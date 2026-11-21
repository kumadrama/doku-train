"""Daily FYP publication and the existing Discovery AppConfig consumer contract."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml
from recommend_tf.inference.artifact_loader import checksum
from recommend_tf.inference.tests.test_idea_publication import (
    RESOURCES,
    AppConfig,
    MemoryS3,
    artifact,
)
from recommend_tf.inference.tests.test_official_pipeline import binding
from recommend_tf.platforms.sagemaker import daily_pipeline, fyp_appconfig
from recommend_tf.platforms.sagemaker.discovery_appconfig import (
    ALIAS,
    BOOTSTRAP_ALIAS,
    ENDPOINT_CALLER,
    LIST_SCHEMA,
    MODEL,
    MODELS_PROFILE,
    SERVICE,
    DiscoveryPublication,
)
from recommend_tf.platforms.sagemaker.idea_appconfig import MODELS_SCHEMA, Publisher, sha256
from recommend_tf.train.comm.export_tasks import DISCOVERY_SERVING_SIGNATURE


def daily_artifact(root, digit):
    candidate = artifact(root, digit * 64)
    (candidate / "feature-state.json").write_text("{}")
    manifest = candidate / "manifest.json"
    value = json.loads(manifest.read_bytes())
    value["files"]["feature-state.json"] = checksum(candidate / "feature-state.json")
    manifest.write_text(json.dumps(value))
    return candidate


class DiscoveryPublicationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        initial = daily_artifact(self.root / "initial", "a")
        listing = {
            "schema_version": LIST_SCHEMA,
            "environment": "dte",
            "models": [
                {
                    "model_name": ALIAS,
                    "model_key": ALIAS,
                    "service_name": SERVICE,
                    "config_name": MODELS_PROFILE,
                },
            ],
        }
        files = {
            "models/initial/" + p.name: {"content": p.read_text(), "sha256": checksum(p)}
            for p in initial.glob("*.json")
        }
        warmup = '{"schema_version":"doku-tf-request/v1","instances":[{}]}'
        files["models/initial/warmup.json"] = {"content": warmup, "sha256": sha256(warmup.encode())}
        models = {
            "schema_version": MODELS_SCHEMA,
            "environment": "dte",
            "inference_dump": {"enabled": False},
            "feature_chains": {"shared": {"retrieval": "fixture"}},
            "models": {
                ALIAS: {
                    "release": {
                        "id": "bootstrap-release",
                        "service": SERVICE,
                        "model": MODEL,
                        "scenes": ["discovery"],
                        "require_dump": True,
                        "endpoint": "doku-dte-fyp-bootstrap",
                        "model_version": "a" * 64,
                        "artifact_dir": "models/initial",
                        "warmup_file": "models/initial/warmup.json",
                        "max_history": 1000,
                        "request_timeout_ms": 2000,
                    },
                    "files": files,
                    "feature_chain": "shared",
                    "model_call": {"type": "sagemaker-tfs/v1"},
                }
            },
        }
        self.api = AppConfig(
            {"list": yaml.safe_dump(listing).encode(), "models": yaml.safe_dump(models).encode()}
        )
        self.options = {
            **binding(),
            "environment": "dte",
            "model_name": MODEL,
            "discovery_appconfig": RESOURCES,
        }
        self.endpoint = self.options["deployment"]["endpoint_name"] = "doku-dte-tf-rerank-official"
        publisher = Publisher(self.api, self.api, RESOURCES, label_prefix="discovery")
        self.driver = DiscoveryPublication(MemoryS3(), publisher, self.options)
        self.base = copy.deepcopy(self.api.live)

    def candidate(self, digit, *, identified=False):
        model = daily_artifact(self.root / digit, digit)
        if identified:
            manifest = model / "manifest.json"
            document = json.loads(manifest.read_bytes())
            document["verification"] = {"serving_version_signature": DISCOVERY_SERVING_SIGNATURE}
            manifest.write_text(json.dumps(document))
        state = SimpleNamespace(
            value={
                "active": {
                    "stage": "uploaded",
                    "operation": digit,
                    "model_version": digit * 64,
                    "model_uri": f"s3://synthetic/{digit}/model.tar.gz",
                }
            },
            save=Mock(),
        )
        return model, state

    def split(self, name):
        options = {
            **self.options,
            "model_name": name,
            "fyp_split_appconfig": RESOURCES,
            "fyp_publication_lock_uri": "s3://synthetic/control/fyp-lock.json",
        }
        endpoint = "doku-dte-fyp-" + name.replace("_", "-")
        options["deployment"] = {**options["deployment"], "endpoint_name": endpoint}
        listing = yaml.safe_load(self.base["list"])
        listing["schema_version"] = fyp_appconfig.LIST_SCHEMA
        routes = listing["models"]
        routes.append(
            {
                "model_name": name,
                "model_key": "split-old",
                "service_name": fyp_appconfig.SERVICE,
                "config_name": "models-config-dte",
            }
        )
        definitions = yaml.safe_load(self.base["models"])
        prior = copy.deepcopy(definitions["models"][ALIAS])
        prior["release"].update(
            id="split-old",
            model=name,
            endpoint=endpoint,
            service=fyp_appconfig.SERVICE,
            scenes=["for_you"],
        )
        definitions["models"]["split-old"] = prior
        stale = copy.deepcopy(prior)
        stale["release"].update(id="split-unserved", model_version="c" * 64)
        definitions["models"]["split-unserved"] = stale
        api = AppConfig(
            {
                "list": yaml.safe_dump(listing).encode(),
                "models": yaml.safe_dump(definitions).encode(),
            }
        )
        driver = fyp_appconfig.SplitFypPublication(
            MemoryS3(),
            Publisher(api, api, RESOURCES, label_prefix="fyp"),
            options,
            ecs=Mock(),
            logs=Mock(),
        )
        return api, driver

    def test_split_cutover_preserves_other_aliases_and_waits_for_replica_load(self):
        model, template = self.candidate("b")
        for name in sorted(fyp_appconfig.SPLIT_MODELS):
            with self.subTest(name=name):
                api, driver = self.split(name)
                original = yaml.safe_load(api.live["list"])
                state = SimpleNamespace(
                    uri="s3://synthetic/control/" + name,
                    value=copy.deepcopy(template.value),
                    save=Mock(),
                )
                self.assertTrue(driver.acquire(state))
                driver.prepare(state, model, self.root)
                self.assertTrue(driver.create_only)
                endpoint = driver.deployment("b" * 64)["endpoint_name"]
                self.assertLessEqual(len(endpoint), 63)
                self.assertTrue(endpoint.endswith("-v" + "b" * 12))
                self.assertEqual(api.starts, 0)
                state.value["active"]["stage"] = driver.endpoint_ready_stage
                driver.rollback_loaded = Mock(side_effect=[False, True])
                self.assertEqual(driver.publish(state), "pending")
                self.assertEqual(driver.publish(state), "deployed")
                self.assertTrue(driver.release(state))
                self.assertEqual(
                    [row["ConfigurationProfileId"] for row in api.deployments[1:]],
                    ["models", "list"],
                )
                routes = json.loads(api.live["list"])["models"]
                self.assertEqual(routes[0], original["models"][0])
                models = json.loads(api.live["models"])["models"]
                self.assertEqual(models[routes[1]["model_key"]]["release"]["endpoint"], endpoint)
                self.assertNotEqual(models["split-old"]["release"]["endpoint"], endpoint)
                self.assertNotIn("split-unserved", models)

    def test_split_readiness_timeout_restores_old_route_once(self):
        api, driver = self.split("inner_watch_lr003")
        old_routes = api.live["list"]
        model, state = self.candidate("b")
        driver.prepare(state, model, self.root)
        state.value["active"]["stage"] = driver.endpoint_ready_stage
        driver.readiness_timeout_seconds = 0
        driver.rollback_loaded = Mock(side_effect=[False, False, True])
        self.assertEqual(driver.publish(state), "pending")
        self.assertEqual(api.live["list"], old_routes)
        self.assertEqual(driver.publish(state), "cutover_rolled_back")
        self.assertEqual(api.starts, 3)

    def test_split_moves_from_pinned_base_to_fixed_endpoint_across_daily_releases(self):
        name = "inner_watch_lr003"
        api, driver = self.split(name)
        base = driver.binding["deployment"]["endpoint_name"]
        driver.rollback_loaded = Mock(return_value=True)

        intermediate, state = self.candidate("b", identified=True)
        driver.prepare(state, intermediate, self.root)
        temporary = driver.deployment_for_state(state.value["active"])["endpoint_name"]
        self.assertEqual(temporary, driver.deployment("b" * 64)["endpoint_name"])
        state.value["active"]["stage"] = driver.endpoint_ready_stage
        self.assertEqual(driver.publish(state), "deployed")
        _, _, _, routed = driver.active_definition(api.live["list"], api.live["models"], "dte")
        self.assertEqual(routed["release"]["model_version"], "b" * 64)

        fixed, state = self.candidate("c", identified=True)
        driver.prepare(state, fixed, self.root)
        self.assertEqual(driver.deployment_for_state(state.value["active"])["endpoint_name"], base)
        _, _, prior_key, _ = driver.active_definition(api.live["list"], api.live["models"], "dte")
        state.value["active"]["stage"] = driver.endpoint_ready_stage
        self.assertEqual(driver.publish(state), "deployed")
        _, definitions, active_key, routed = driver.active_definition(
            api.live["list"], api.live["models"], "dte"
        )
        self.assertEqual(routed["model_call"]["type"], ENDPOINT_CALLER)
        self.assertNotIn("model_version", routed["release"])
        self.assertEqual(definitions["models"][prior_key]["release"]["endpoint"], temporary)
        after = copy.deepcopy(api.live)

        candidate, state = self.candidate("d", identified=True)
        state.uri = "s3://synthetic/control/split.json"
        state.value["online"] = {
            "endpoint_config": "previous",
            "endpoint_name": base,
            "model_version": "c" * 64,
        }
        driver.prepare(state, candidate, self.root)
        journal = state.value["active"][driver.journal_key]
        self.assertEqual(journal["candidate_release"], active_key)
        self.assertEqual(journal["candidate_version"], "d" * 64)
        sm, runtime = Mock(), Mock()
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": "previous",
        }
        sm.describe_model.return_value = None
        sm.describe_endpoint_config.return_value = None

        def response(**call):
            outputs = dict.fromkeys(driver.binding["outputs"], [0.5])
            if json.loads(call["Body"])["signature_name"] == DISCOVERY_SERVING_SIGNATURE:
                self.assertNotIn("CustomAttributes", call)
                outputs["__doku_model_version"] = ["d" * 64]
            else:
                self.assertIn("CustomAttributes", call)
            return {"Body": io.BytesIO(json.dumps({"outputs": outputs}).encode())}

        runtime.invoke_endpoint.side_effect = response
        args = (driver.s3, sm, runtime, state, driver.binding)
        kwargs = {"work": self.root, "config": {}, "model_publication": driver}
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        self.assertEqual(sm.update_endpoint.call_args.kwargs["EndpointName"], base)
        sm.create_endpoint.assert_not_called()
        sm.describe_endpoint.return_value["EndpointConfigName"] = state.value["active"][
            "endpoint_config"
        ]
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        self.assertEqual(runtime.invoke_endpoint.call_count, 2)
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "deployed")
        self.assertEqual(api.live, after)
        self.assertEqual(api.starts, 4)

        candidate, rejected = self.candidate("e", identified=True)
        rejected.uri = state.uri
        previous_config = state.value["active"]["endpoint_config"]
        rejected.value["online"] = {
            "endpoint_config": previous_config,
            "endpoint_name": base,
            "model_version": "d" * 64,
        }
        driver.prepare(rejected, candidate, self.root)
        self.assertEqual(rejected.value["active"][driver.journal_key]["previous_version"], "d" * 64)
        self.assertTrue(driver.release(state))
        args = (driver.s3, sm, runtime, rejected, driver.binding)
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        sm.describe_endpoint.return_value["EndpointConfigName"] = rejected.value["active"][
            "endpoint_config"
        ]
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        self.assertEqual(rejected.value["active"]["stage"], "rollback_pending")
        self.assertEqual(sm.update_endpoint.call_args.kwargs["EndpointName"], base)
        self.assertEqual(sm.update_endpoint.call_args.kwargs["EndpointConfigName"], previous_config)
        sm.describe_endpoint.return_value["EndpointConfigName"] = previous_config
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "rolled_back")
        self.assertEqual(api.live, after)

    def test_shared_publication_lock_is_exclusive_and_resumes_its_owner(self):
        _, driver = self.split("inner_watch_lr003")
        _, first = self.candidate("b")
        _, second = self.candidate("c")
        first.uri = "s3://synthetic/control/first.json"
        second.uri = "s3://synthetic/control/second.json"
        self.assertTrue(driver.acquire(first))
        self.assertFalse(driver.acquire(second))
        self.assertTrue(driver.release(second))
        self.assertFalse(driver.acquire(second))
        resumed = SimpleNamespace(uri=first.uri, value=copy.deepcopy(first.value), save=Mock())
        self.assertTrue(driver.acquire(resumed))
        self.assertTrue(driver.release(resumed))
        self.assertTrue(driver.acquire(second))
        self.assertTrue(driver.release(resumed))
        self.assertTrue(driver.acquire(second))

    def test_joint_fyp_publication_waits_for_all_replicas_before_shared_endpoint_rotation(self):
        route = yaml.safe_load(self.api.live["list"])
        route["schema_version"] = fyp_appconfig.LIST_SCHEMA
        route["models"][0].update(
            model_name=MODEL, service_name=fyp_appconfig.SERVICE, config_name="models-config-dte"
        )
        definitions = yaml.safe_load(self.api.live["models"])
        definitions["models"][ALIAS]["release"].update(
            service=fyp_appconfig.SERVICE, scenes=["for_you"]
        )
        definitions["training_dump"] = {"delivery_stream": "fyp-only"}
        resources = {
            **RESOURCES,
            "model_list_profile": "fyp-list",
            "models_config_profile": "fyp-models",
        }
        api = AppConfig(
            {
                **self.api.live,
                "fyp-list": yaml.safe_dump(route).encode(),
                "fyp-models": yaml.safe_dump(definitions).encode(),
            }
        )
        s3 = MemoryS3()
        options = {**self.options, "fyp_appconfig": resources}
        both = fyp_appconfig.FypAndDiscoveryPublication(
            s3,
            Publisher(api, api, RESOURCES, label_prefix="discovery"),
            options,
        )
        both.fyp = fyp_appconfig.FypPublication(
            s3,
            Publisher(api, api, resources, label_prefix="fyp"),
            options,
            ecs=Mock(),
            logs=Mock(),
        )
        model, state = self.candidate("b")
        both.prepare(state, model, self.root)
        self.assertTrue(both.before_endpoint(state))
        state.value["active"]["stage"] = both.endpoint_ready_stage
        self.assertEqual(both.publish(state), "deployed")
        self.assertEqual(both.publish(state), "deployed")
        self.assertEqual(api.starts, 4)
        fyp_routes = json.loads(api.live["fyp-list"])["models"]
        fyp_models = json.loads(api.live["fyp-models"])
        self.assertEqual(fyp_models["training_dump"], definitions["training_dump"])
        self.assertEqual(
            fyp_models["models"][fyp_routes[0]["model_key"]]["release"]["model_version"], "b" * 64
        )
        self.assertEqual(fyp_routes[1]["model_name"], fyp_appconfig.ROLLBACK)
        self.assertEqual(
            fyp_models["models"][fyp_routes[1]["model_key"]]["release"]["id"], "bootstrap-release"
        )
        self.assertEqual(yaml.safe_load(api.live["models"])["inference_dump"], {"enabled": False})

        model, state = self.candidate("c", identified=True)
        both.prepare(state, model, self.root)
        both.fyp.rollback_loaded = Mock(side_effect=[False, True])
        sm, runtime = Mock(), Mock()
        args = (s3, sm, runtime, state, options)
        kwargs = {"work": self.root, "config": {}, "model_publication": both}
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        sm.describe_endpoint.assert_not_called()
        rollback_routes = json.loads(api.live["fyp-list"])["models"]
        self.assertEqual(rollback_routes[0]["model_key"], rollback_routes[1]["model_key"])
        self.assertTrue(both.before_endpoint(state))
        self.assertEqual(api.starts, 5)
        state.value["active"]["stage"] = both.endpoint_ready_stage
        self.assertEqual(both.publish(state), "deployed")
        self.assertEqual(api.starts, 9)
        next_routes = json.loads(api.live["fyp-list"])["models"]
        next_models = json.loads(api.live["fyp-models"])["models"]
        active_model = next_models[next_routes[0]["model_key"]]
        self.assertEqual(active_model["model_call"]["type"], ENDPOINT_CALLER)
        self.assertNotIn("model_version", active_model["release"])
        self.assertEqual(
            next_models[next_routes[1]["model_key"]]["release"]["model_version"], "a" * 64
        )
        live = copy.deepcopy(api.live)
        model, state = self.candidate("d", identified=True)
        state.value["online"] = {
            "endpoint_config": "previous",
            "endpoint_name": self.endpoint,
            "model_version": "c" * 64,
        }
        both.prepare(state, model, self.root)
        self.assertTrue(both.before_endpoint(state))
        self.assertEqual(api.live, live)
        state.value["active"]["stage"] = both.endpoint_ready_stage
        self.assertEqual(both.publish(state), "deployed")
        self.assertEqual(api.live, live)
        self.assertEqual(api.starts, 9)

    def test_three_daily_models_publish_definitions_then_routes_to_existing_endpoint(self):
        for digit in ("b", "c", "d"):
            model, state = self.candidate(digit)
            previous = self.api.live["list"]
            self.driver.prepare(state, model, self.root)
            with self.assertRaises(ValueError):
                self.driver.publish(state)
            self.assertEqual(self.api.live["list"], previous)
            state.value["active"]["stage"] = self.driver.endpoint_ready_stage
            self.assertEqual(self.driver.publish(state), "deployed")
            self.assertEqual(self.driver.publish(state), "deployed")
            self.assertEqual(
                [x["ConfigurationProfileId"] for x in self.api.deployments[-2:]], ["models", "list"]
            )
            defs = json.loads(self.api.live["models"])["models"]
            routes = json.loads(self.api.live["list"])["models"]
            self.assertEqual([r["model_name"] for r in routes], [ALIAS, BOOTSTRAP_ALIAS])
            self.assertTrue(all(r["model_key"] in defs for r in yaml.safe_load(previous)["models"]))
            self.assertLessEqual(len(defs), 3)
            release = defs[routes[0]["model_key"]]["release"]
            self.assertEqual(release["model_version"], digit * 64)
            self.assertEqual(release["endpoint"], self.endpoint)
            self.assertEqual(release["service"], SERVICE)
            self.assertEqual(release["scenes"], ["discovery"])
            self.assertTrue(release["require_dump"])
            self.assertEqual(release["max_history"], 1000)
            self.assertEqual(defs[routes[1]["model_key"]]["release"]["id"], "bootstrap-release")
        self.assertEqual(self.api.starts, 6)

    def test_same_daily_release_is_a_noop_and_failed_contract_cannot_publish(self):
        model, state = self.candidate("b")
        self.driver.prepare(state, model, self.root)
        state.value["active"]["stage"] = self.driver.endpoint_ready_stage
        self.driver.publish(state)
        repeat = SimpleNamespace(
            value={"active": {k: v for k, v in state.value["active"].items() if k != "online"}},
            save=Mock(),
        )
        repeat.value["active"]["stage"] = "uploaded"
        self.driver.prepare(repeat, model, self.root)
        repeat.value["active"]["stage"] = self.driver.endpoint_ready_stage
        self.assertEqual(self.driver.publish(repeat), "deployed")
        self.assertEqual(self.api.starts, 2)
        model, state = self.candidate("c")
        (model / "feature-config.json").write_text('{"invalid":"contract"}')
        with self.assertRaises(ValueError):
            self.driver.prepare(state, model, self.root)
        self.assertNotIn("online", state.value["active"])
        self.assertEqual(self.api.starts, 2)

    def test_identified_discovery_activates_once_on_the_fixed_endpoint(self):
        for digit in ("b", "c"):
            model, state = self.candidate(digit, identified=True)
            self.driver.prepare(state, model, self.root)
            state.value["active"]["stage"] = self.driver.endpoint_ready_stage
            self.assertEqual(self.driver.publish(state), "deployed")
            if digit == "b":
                route = json.loads(self.api.live["list"])["models"][0]
                definition = json.loads(self.api.live["models"])["models"][route["model_key"]]
                self.assertEqual(definition["model_call"]["type"], ENDPOINT_CALLER)
                self.assertNotIn("model_version", definition["release"])
                self.assertEqual(definition["release"]["endpoint"], self.endpoint)
                self.assertEqual(
                    json.loads(self.api.live["list"])["models"][1]["model_name"], BOOTSTRAP_ALIAS
                )
                enabled_documents = copy.deepcopy(self.api.live)
            else:
                self.assertEqual(self.api.live, enabled_documents)

        legacy, state = self.candidate("d")
        with self.assertRaisesRegex(ValueError, "requires the model identity signature"):
            self.driver.prepare(state, legacy, self.root)
        self.assertEqual(self.api.starts, 2)

    def test_rolled_back_definitions_and_lost_response_keep_replay_safe(self):
        model, state = self.candidate("b")
        self.driver.prepare(state, model, self.root)
        state.value["active"]["stage"] = self.driver.endpoint_ready_stage
        self.api.lose_create = True
        with self.assertRaises(TimeoutError):
            self.driver.publish(state)
        self.api.lose_start = True
        with self.assertRaises(TimeoutError):
            self.driver.publish(state)
        self.assertEqual(self.driver.publish(state), "deployed")
        self.assertEqual(self.api.starts, 2)
        model, state = self.candidate("c")
        previous = self.api.live["list"]
        self.driver.prepare(state, model, self.root)
        state.value["active"]["stage"] = self.driver.endpoint_ready_stage
        self.api.auto_rollback.add("models")
        with self.assertRaises(ValueError):
            self.driver.publish(state)
        self.assertEqual(self.api.live["list"], previous)
        self.assertEqual(state.value["active"]["stage"], self.driver.endpoint_ready_stage)

    def test_daily_pipeline_waits_for_endpoint_probe_then_publishes(self):
        model, state = self.candidate("b")
        self.driver.prepare(state, model, self.root)
        sm = Mock()
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": "previous",
        }
        sm.describe_model.return_value = None
        sm.describe_endpoint_config.return_value = None
        runtime = Mock()
        kwargs = {"work": self.root, "config": {}, "model_publication": self.driver}
        args = (self.driver.s3, sm, runtime, state, self.options)
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        self.assertEqual(sm.update_endpoint.call_args.kwargs["EndpointName"], self.endpoint)
        sm.create_endpoint.assert_not_called()
        self.assertEqual(self.api.starts, 0)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": state.value["active"]["endpoint_config"],
        }
        runtime.invoke_endpoint.return_value = {
            "Body": io.BytesIO(
                json.dumps({"outputs": dict.fromkeys(self.options["outputs"], [0.5])}).encode()
            )
        }
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "pending")
        self.assertEqual(self.api.starts, 0)
        self.assertEqual(daily_pipeline.advance_release(*args, **kwargs), "deployed")
        self.assertEqual(self.api.starts, 2)
        sm.create_training_job.assert_not_called()
