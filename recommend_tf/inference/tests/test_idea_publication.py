"""Version endpoint reuse and guarded Idea hot publication; no cloud calls."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from recommend_tf.inference.artifact_loader import checksum
from recommend_tf.inference.tests.test_official import fixture
from recommend_tf.inference.tests.test_official_pipeline import binding
from recommend_tf.platforms.sagemaker.idea_appconfig import Publisher, encode, sha256
from recommend_tf.platforms.sagemaker.idea_publication import IdeaPublication, deployment_for
from recommend_tf.platforms.sagemaker.publish_model import deploy_step


class APIError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class MemoryS3:
    def __init__(self):
        self.data = {}
        self.number = 0

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.data:
            raise APIError("404")
        raw, etag = self.data[Bucket, Key]
        return {"Body": io.BytesIO(raw), "ETag": etag, "ContentLength": len(raw)}

    def put_object(self, Bucket, Key, Body, **kwargs):
        old = self.data.get((Bucket, Key))
        if kwargs.get("IfNoneMatch") == "*" and old:
            raise APIError("PreconditionFailed")
        if kwargs.get("IfMatch") and (not old or old[1] != kwargs["IfMatch"]):
            raise APIError("PreconditionFailed")
        self.number += 1
        raw = Body.read() if hasattr(Body, "read") else Body
        etag = str(self.number)
        self.data[Bucket, Key] = raw, etag
        return {"ETag": etag}


class AppConfig:
    def __init__(self, profiles):
        self.live = dict(profiles)
        self.versions = {
            key: [{"VersionNumber": 1, "VersionLabel": "initial", "raw": raw}]
            for key, raw in profiles.items()
        }
        self.deployments = [
            {"DeploymentNumber": 10, "VersionLabel": "initial", "State": "COMPLETE"}
        ]
        self.starts = 0
        self.lose_start = False
        self.lose_create = False
        self.race = False
        self.auto_rollback = set()

    def start_configuration_session(self, **kwargs):
        return {"InitialConfigurationToken": kwargs["ConfigurationProfileIdentifier"]}

    def get_latest_configuration(self, ConfigurationToken):
        return {"Configuration": io.BytesIO(self.live[ConfigurationToken])}

    def list_hosted_configuration_versions(self, ConfigurationProfileId, **kwargs):
        return {
            "Items": [
                dict(item)
                for item in self.versions[ConfigurationProfileId]
                if not kwargs.get("VersionLabel") or item["VersionLabel"] == kwargs["VersionLabel"]
            ]
        }

    def get_hosted_configuration_version(self, ConfigurationProfileId, VersionNumber, **kwargs):
        raw = next(
            item["raw"]
            for item in self.versions[ConfigurationProfileId]
            if item["VersionNumber"] == VersionNumber
        )
        return {"Content": io.BytesIO(raw)}

    def create_hosted_configuration_version(
        self, ConfigurationProfileId, LatestVersionNumber, Content, VersionLabel, **kwargs
    ):
        values = self.versions[ConfigurationProfileId]
        if max(item["VersionNumber"] for item in values) != LatestVersionNumber:
            raise APIError("ConflictException")
        item = {
            "VersionNumber": LatestVersionNumber + 1,
            "VersionLabel": VersionLabel,
            "raw": Content,
        }
        values.append(item)
        if self.lose_create:
            self.lose_create = False
            raise TimeoutError("lost create response")
        return item

    def list_deployments(self, **kwargs):
        return {"Items": list(reversed(self.deployments))}

    def start_deployment(
        self, ConfigurationProfileId, ConfigurationVersion, LatestDeploymentNumber, **kwargs
    ):
        if self.race:
            self.race = False
            self.deployments.append(
                {
                    "DeploymentNumber": self.deployments[-1]["DeploymentNumber"] + 1,
                    "VersionLabel": "external",
                    "State": "COMPLETE",
                }
            )
            self.live[ConfigurationProfileId] = b"external"
        if self.deployments[-1]["DeploymentNumber"] != LatestDeploymentNumber:
            raise APIError("ConflictException")
        item = next(
            item
            for item in self.versions[ConfigurationProfileId]
            if str(item["VersionNumber"]) == ConfigurationVersion
        )
        result = {
            "DeploymentNumber": LatestDeploymentNumber + 1,
            "VersionLabel": item["VersionLabel"],
            "ConfigurationProfileId": ConfigurationProfileId,
            "ConfigurationVersion": ConfigurationVersion,
            "State": "COMPLETE",
        }
        if ConfigurationProfileId in self.auto_rollback:
            result["State"] = "ROLLED_BACK"
        else:
            self.live[ConfigurationProfileId] = item["raw"]
        self.deployments.append(result)
        self.starts += 1
        if self.lose_start:
            self.lose_start = False
            raise TimeoutError("lost start response")
        return result

    def get_deployment(self, DeploymentNumber, **kwargs):
        return next(
            item for item in self.deployments if item["DeploymentNumber"] == DeploymentNumber
        )


RESOURCES = {
    "application_id": "app",
    "environment_id": "env",
    "deployment_strategy_id": "strategy",
    "model_list_profile": "list",
    "models_config_profile": "models",
}


class PublisherTests(unittest.TestCase):
    def setup(self):
        api = AppConfig({"list": b"before"})
        publisher = Publisher(api, api, RESOURCES)
        state = SimpleNamespace(
            value={"active": {"operation": "operation", "online": {}}}, save=lambda: None
        )
        return api, publisher, state

    def test_crash_before_start_replays_same_server_cas(self):
        api, publisher, state = self.setup()

        def save():
            if state.value["active"]["online"]["publications"]["route"].get("start_attempted"):
                raise RuntimeError("crashed before API")

        state.save = save
        with self.assertRaises(RuntimeError):
            publisher.step(state, "route", "list", b"before", b"after")
        state.save = lambda: None
        self.assertTrue(publisher.step(state, "route", "list", b"before", b"after"))
        self.assertEqual(api.starts, 1)

    def test_lost_api_responses_reconcile_without_duplicate_mutations(self):
        for flag in ("lose_create", "lose_start"):
            api, publisher, state = self.setup()
            setattr(api, flag, True)
            with self.assertRaises(TimeoutError):
                publisher.step(state, "route", "list", b"before", b"after")
            self.assertTrue(publisher.step(state, "route", "list", b"before", b"after"))
            self.assertEqual(len(api.versions["list"]), 2)
            self.assertEqual(api.starts, 1)

    def test_last_moment_external_deployment_is_not_overwritten(self):
        api, publisher, state = self.setup()
        api.race = True
        with self.assertRaises(APIError):
            publisher.step(state, "route", "list", b"before", b"after")
        self.assertEqual(api.live["list"], b"external")
        with self.assertRaises(ValueError):
            publisher.step(state, "route", "list", b"before", b"after")
        self.assertEqual(api.starts, 0)

    def test_changed_companion_and_unsupported_sdk_fail_closed(self):
        api, publisher, state = self.setup()
        with self.assertRaises(ValueError):
            publisher.step(
                state,
                "route",
                "list",
                b"before",
                b"after",
                companion=lambda: (_ for _ in ()).throw(ValueError("changed")),
            )
        self.assertEqual(len(api.versions["list"]), 1)
        api.meta = SimpleNamespace(
            service_model=SimpleNamespace(
                operation_model=lambda name: SimpleNamespace(
                    input_shape=SimpleNamespace(members={})
                )
            )
        )
        with self.assertRaises(ValueError):
            Publisher(api, api, RESOURCES)


def artifact(root, version):
    source = fixture(root)
    signature = json.loads((source / "signature.json").read_text())
    signature["model_version"] = version
    (source / "signature.json").write_text(json.dumps(signature))
    (source / "saved_model/saved_model.pb").write_bytes(("synthetic model " + version).encode())
    manifest = {
        "version": "doku-tf-inference/v1",
        "model_version": version,
        "files": {
            p.relative_to(source).as_posix(): checksum(p)
            for p in source.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    return source


def documents(source):
    files = {}
    for path in source.glob("*.json"):
        raw = path.read_bytes()
        files["previous/" + path.name] = {"sha256": sha256(raw), "content": raw.decode()}
    raw = b"synthetic DAG"
    files["previous/features.yml"] = {"sha256": sha256(raw), "content": raw.decode()}
    definition = {
        "release": {
            "id": "old-release",
            "model": "mapped",
            "service": "idea_tab_rank",
            "artifact_dir": "previous",
            "model_version": "a" * 64,
            "manifest_sha256": checksum(source / "manifest.json"),
            "max_history": 1024,
            "batch_size": 256,
            "request_timeout_ms": 450,
        },
        "feature_chain": "idea-chain",
        "model_call": {"type": "sagemaker-tfs/v1"},
        "files": files,
        "provenance": {"source": "s3://synthetic/old/model.tar.gz"},
    }
    listing = {
        "schema_version": "ai-engine/idea-model-list/v1",
        "environment": "dte",
        "reload_interval_ms": 30000,
        "models": [
            {
                "model_name": "mapped",
                "model_key": "old-release",
                "service_name": "idea_tab_rank",
                "config_name": "idea_models_config",
            }
        ],
    }
    models = {
        "schema_version": "ai-engine/models-config/v1",
        "environment": "dte",
        "feature_chains": {"idea-chain": {"retrieval": {"type": "dag/v1"}}},
        "models": {"old-release": definition, "expired": {"unused": True}},
        "inference_dump": {"enabled": True},
    }
    return encode(listing), encode(models)


class IdeaPublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        old, self.new = artifact(self.root / "old", "a" * 64), artifact(self.root / "new", "b" * 64)
        listing, models = documents(old)
        value = json.loads(models)
        value["models"]["old-release"]["release"]["endpoint"] = "idea-v-" + "a" * 12
        self.api = AppConfig({"list": listing, "models": encode(value)})
        self.options = binding()
        self.options.update(
            environment="dte",
            data_source={"freeze": "idea_mapped"},
            idea_appconfig={**RESOURCES, "endpoint_prefix": "idea-v-"},
        )
        self.state = SimpleNamespace(
            value={
                "active": {
                    "stage": "uploaded",
                    "operation": "op",
                    "model_version": "b" * 64,
                    "model_uri": "s3://synthetic/models/new.tar.gz",
                }
            },
            save=Mock(),
        )
        self.publisher = Publisher(self.api, self.api, RESOURCES)
        self.driver = IdeaPublication(MemoryS3(), self.publisher, self.options)

    def test_version_endpoint_and_documents_keep_previous_identity(self):
        selected = deployment_for(self.options, "b" * 64)
        self.assertEqual(selected["endpoint_name"], "idea-v-" + "b" * 12)
        self.assertEqual(self.options["deployment"]["endpoint_name"], "synthetic")
        original = binding()
        self.assertIs(deployment_for(original, "irrelevant"), original["deployment"])
        self.driver.prepare(self.state, self.new, self.root)
        self.state.value["active"]["stage"] = "idea_endpoint_ready"
        self.assertEqual(self.driver.publish(self.state), "deployed")
        self.assertEqual(
            [v["ConfigurationProfileId"] for v in self.api.deployments[1:]], ["models", "list"]
        )
        models = json.loads(self.api.live["models"])
        route = json.loads(self.api.live["list"])["models"][0]
        self.assertEqual(len(json.loads(self.api.live["list"])["models"]), 1)
        candidate = models["models"][route["model_key"]]["release"]
        self.assertEqual(candidate["endpoint"], "idea-v-" + "b" * 12)
        self.assertEqual(candidate["max_history"], 1024)
        self.assertEqual(candidate["request_timeout_ms"], 450)
        self.assertEqual(
            models["models"]["old-release"]["release"]["endpoint"], "idea-v-" + "a" * 12
        )
        self.assertEqual(set(models["models"]), {"old-release", route["model_key"]})

    def test_release_receipt_records_actual_endpoint_with_legacy_fallback(self):
        from recommend_tf.platforms.monitoring.report_metrics import release_event

        options = {**self.options, "model_name": "idea_mapped"}
        record = {
            "stage": "deployed",
            "operation": "op",
            "release_revision": 1,
            "finished_at_ms": 1,
            "model_version": "b" * 64,
            "endpoint_config": "config",
            "endpoint_name": "idea-v-" + "b" * 12,
        }
        event = release_event("s3://synthetic/state", record, options)
        self.assertEqual(event["deployment"]["endpoint_name"], record["endpoint_name"])
        record.pop("endpoint_name")
        event = release_event("s3://synthetic/state", record, options)
        self.assertEqual(
            event["deployment"]["endpoint_name"], options["deployment"]["endpoint_name"]
        )

    def test_reentry_does_not_repeat_hot_publication(self):
        self.driver.prepare(self.state, self.new, self.root)
        self.state.value["active"]["stage"] = "idea_endpoint_ready"
        self.driver.publish(self.state)
        self.driver.publish(self.state)
        self.assertEqual(self.api.starts, 2)

    def test_external_route_change_is_preserved(self):
        self.driver.prepare(self.state, self.new, self.root)
        self.state.value["active"]["stage"] = "idea_endpoint_ready"
        self.api.live["list"] = b"external"
        with self.assertRaises(ValueError):
            self.driver.publish(self.state)
        self.assertEqual(self.api.starts, 0)
        self.assertEqual(self.api.live["list"], b"external")

    def test_failed_definition_does_not_touch_old_route(self):
        before = self.api.live["list"]
        self.driver.prepare(self.state, self.new, self.root)
        self.state.value["active"]["stage"] = "idea_endpoint_ready"
        self.api.auto_rollback.add("models")
        with self.assertRaises(ValueError):
            self.driver.publish(self.state)
        self.assertEqual(self.api.live["list"], before)
        self.assertEqual(self.state.value["active"]["stage"], "idea_endpoint_ready")

    def test_changed_signature_fails_before_configuration_writes(self):
        changed = json.loads((self.new / "signature.json").read_text())
        changed["inputs"]["extra"] = {}
        (self.new / "signature.json").write_text(json.dumps(changed))
        manifest = json.loads((self.new / "manifest.json").read_text())
        manifest["files"]["signature.json"] = checksum(self.new / "signature.json")
        (self.new / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.driver.prepare(self.state, self.new, self.root)
        self.state.value["active"]["stage"] = "idea_endpoint_ready"
        self.assertEqual(self.driver.s3.data, {})


class VersionEndpointTest(unittest.TestCase):
    def test_create_only_never_updates_other_endpoint_or_rolls_it_back(self):
        options = binding()["deployment"]
        state = SimpleNamespace(
            value={
                "active": {
                    "stage": "uploaded",
                    "model_version": "a" * 64,
                    "model_uri": "s3://synthetic/model.tar.gz",
                }
            },
            save=Mock(),
        )
        sm = Mock()
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": "external",
        }
        with self.assertRaises(ValueError):
            deploy_step(sm, None, state, options, probe=Mock(), create_only=True)
        sm.create_model.assert_not_called()
        sm.update_endpoint.assert_not_called()

        # Use the existing deployment state machine to create a fresh endpoint.
        sm.describe_endpoint.return_value = None
        sm.describe_model.return_value = None
        sm.describe_endpoint_config.return_value = None
        self.assertEqual(
            deploy_step(
                sm,
                None,
                state,
                options,
                probe=Mock(),
                create_only=True,
                completion_stage="idea_endpoint_ready",
            ),
            "pending",
        )
        sm.create_endpoint.assert_called_once()
        sm.update_endpoint.assert_not_called()
        name = state.value["active"]["endpoint_config"]
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": name,
        }
        state.value["active"]["previous_config"] = name

        def failed(*args):
            raise ValueError("synthetic failed model")

        self.assertEqual(
            deploy_step(sm, None, state, options, probe=failed, create_only=True),
            "deployment_failed",
        )
        sm.update_endpoint.assert_not_called()

    def test_original_endpoint_completion_remains_default(self):
        options = binding()["deployment"]
        sm = Mock()
        sm.describe_model.return_value = None
        sm.describe_endpoint_config.return_value = None
        sm.describe_endpoint.return_value = None
        state = SimpleNamespace(
            value={
                "active": {
                    "stage": "uploaded",
                    "model_version": "a" * 64,
                    "model_uri": "s3://synthetic/model.tar.gz",
                }
            },
            save=Mock(),
        )
        deploy_step(sm, None, state, options, probe=Mock())
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "EndpointConfigName": state.value["active"]["endpoint_config"],
        }
        self.assertEqual(deploy_step(sm, None, state, options, probe=Mock()), "deployed")


if __name__ == "__main__":
    unittest.main()
