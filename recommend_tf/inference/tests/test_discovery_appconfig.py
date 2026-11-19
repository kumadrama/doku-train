"""Daily FYP publication and the existing Discovery AppConfig consumer contract."""

import copy
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
    LIST_SCHEMA,
    MODEL,
    MODELS_PROFILE,
    SERVICE,
    DiscoveryPublication,
)
from recommend_tf.platforms.sagemaker.idea_appconfig import MODELS_SCHEMA, Publisher, sha256


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

    def candidate(self, digit):
        model = daily_artifact(self.root / digit, digit)
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

        model, state = self.candidate("c")
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
        self.assertEqual(
            next_models[next_routes[0]["model_key"]]["release"]["model_version"], "c" * 64
        )
        self.assertEqual(
            next_models[next_routes[1]["model_key"]]["release"]["model_version"], "a" * 64
        )

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
        import io

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
