import copy
import hashlib
import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import tensorflow.compat.v1 as tf
from botocore.exceptions import ClientError

from recommend_tf.platforms.sagemaker.publish_model import (
    ReleaseState,
    deploy_step,
    publish_package,
)
from recommend_tf.train.comm.datasvr.daily_contracts import digest
from recommend_tf.train.comm.datasvr.daily_dataset import freeze, resolve_config, window
from recommend_tf.train.daily_train_model import execute as daily
from recommend_tf.train.experiment import execute as experiment
from recommend_tf.train.export_model import execute as export
from recommend_tf.train.models.rerank.data_utils import migrate_experiment
from recommend_tf.train.tests.experiment_fixture import make_experiment


def fixture(root):
    source = make_experiment(root / "fixture", "deepfm", rows_per_day=4)
    source["max_epochs"] = 1
    schema = pq.read_schema(root / "fixture/dataset/day-0.parquet")
    dtypes = {
        f.name: "str"
        if str(f.type) == "string"
        else "int64"
        if "int64" in str(f.type)
        else "float32"
        for f in schema
        if f.name != "impression_time"
    }
    config = migrate_experiment(
        source,
        dtypes,
        {"optimizer": "adamw", "learning_rate": 0.004, "optimizer_args": {"weight_decay": 0.0001}},
        16,
    )
    config["model_name"] = "synthetic_rerank"
    config["online_config"] = {"training_date": 6, "validation_date": 1}
    dataset = root / "fixture/dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    contracts = {
        k: (dataset / "contracts" / (k + ".json")).read_bytes() for k in ("features", "labels")
    }
    sources = []
    for part in manifest["partitions"][:7]:
        day = part["event_time_from"][:10]
        folder = root / day
        (folder / "contracts").mkdir(parents=True)
        for k, body in contracts.items():
            (folder / "contracts" / (k + ".json")).write_bytes(body)
        one = dict(
            manifest,
            partitions=[part],
            row_count=part["rows"],
            label_config_version="fixture-label/v1",
            projection_sha256=digest(json.loads(contracts["features"])),
        )
        one["contract_sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in contracts.items()}
        one["content_sha256"] = digest({k: v for k, v in one.items() if k != "content_sha256"})
        body = json.dumps(one).encode()
        (folder / "manifest.json").write_bytes(body)
        sources.append(
            {
                "date": day,
                "manifest_uri": (folder / "manifest.json").as_uri(),
                "manifest_sha256": hashlib.sha256(body).hexdigest(),
                "contract_sha256": one["contract_sha256"],
                "status": "published",
                "maturity_evidence": "synthetic-ready/v1",
            }
        )
    return config, {
        "version": "doku-tf-dataset-ready/v1",
        "timezone": "UTC",
        "cutoff": "2026-10-07",
        "sources": sources,
    }


def test_window_and_missing_ready(tmp_path):
    assert window({"online_config": {"training_date": 2, "validation_date": 1}}, "2026-01-01") == {
        "train": ["2025-12-30", "2025-12-31"],
        "validation": ["2026-01-01"],
    }
    for bad in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            window({"online_config": {"training_date": bad, "validation_date": 1}}, "2026-01-01")
    config, ready = fixture(tmp_path)
    ready["sources"].pop()
    with pytest.raises(ValueError, match="exact contiguous"):
        freeze(config, ready, tmp_path / "frozen")
    assert "test_from" not in resolve_config(config, "2026-10-07")["dataset_config"]["split"]


def test_daily_reuses_training_and_exports_without_labels(tmp_path):
    config, ready = fixture(tmp_path)
    frozen = tmp_path / "frozen"
    resolved = freeze(config, ready, frozen)
    daily_root = tmp_path / "daily"
    status = daily(resolved, daily_root, cutoff=ready["cutoff"], frozen_input=frozen / "input")
    assert status["status"] == "completed"
    assert not (daily_root / "prepared/test").exists()
    assert not (daily_root / "test").exists()
    # Compare to unchanged three-stage experiment with the identical train/validation.
    offline = copy.deepcopy(config)
    offline_root = tmp_path / "offline"
    experiment(offline, offline_root)
    a, b = [
        json.loads((p / "training/selection.json").read_text()) for p in (daily_root, offline_root)
    ]
    assert a["epoch"] == b["epoch"] and a["score"] == pytest.approx(b["score"], abs=1e-7)
    readers = [
        tf.train.load_checkpoint(str(p / "training/epoch-0001/model"))
        for p in (daily_root, offline_root)
    ]
    for name in readers[0].get_variable_to_shape_map():
        np.testing.assert_array_equal(readers[0].get_tensor(name), readers[1].get_tensor(name))
    outputs = {
        "completion": "probability",
        "fast_swipe": "probability",
        "immersive_click": "probability",
        "watch_seconds": "seconds",
    }
    package = tmp_path / "package"
    result = export(daily_root, package, outputs=outputs)
    signature = json.loads((package / "signature.json").read_text())
    assert set(signature["outputs"]) == set(outputs)
    features = json.loads((package / "feature-config.json").read_text())
    assert "video_finish_seen" not in features["dtypes"]
    assert "watch_duration_ms" not in features["defaults"]
    assert signature["model_version"] == result["model_version"]
    assert not any(
        "watch_duration_ms" in key or "video_finish_seen" in key for key in signature["inputs"]
    )
    with pytest.raises(FileExistsError):
        export(daily_root, package, outputs=outputs)
    second = tmp_path / "package-again"
    export(daily_root, second, outputs=outputs)
    assert (package / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    # Exercise restartable platform export in a fresh process from downloaded sealed files.
    from recommend_tf.platforms.sagemaker.daily_pipeline import step

    s3 = S3()
    evidence = json.loads((daily_root / "COMMITTED.json").read_text())
    for name in [*evidence["files"], "COMMITTED.json"]:
        s3.put_object(Bucket="bucket", Key="run/" + name, Body=(daily_root / name).read_bytes())
    state = ReleaseState(s3, "s3://bucket/state", "key")
    active = state.claim("bootstrap", "2026-10-07")
    active.update(stage="trained", run_uri="s3://bucket/run")
    state.save()
    _, sm, deployment = deployment_fixture()
    binding = {
        "download_limit_bytes": 100_000_000,
        "export_timeout_seconds": 60,
        "outputs": outputs,
        "model_prefix": "s3://bucket/models",
        "kms_key": "key",
        "deployment": deployment,
        "probe": {},
    }
    work = tmp_path / "platform"
    work.mkdir()
    assert step(s3, sm, None, state, {}, {}, binding, work=work) == "pending"
    assert active["stage"] == "deploying" and active["model_version"] == result["model_version"]
    assert active["export_manifest_sha256"] == result["manifest_sha256"]
    from unittest.mock import patch

    from recommend_tf.train.comm.artifact_integrity import seal_run

    with patch(
        "recommend_tf.train.export_model.exporter_identity", return_value="updated-exporter"
    ):
        updated = export(daily_root, tmp_path / "updated-export", outputs=outputs)
    assert updated["model_version"] != result["model_version"]
    config_path = daily_root / "experiment-config.json"
    original = config_path.read_bytes()
    mixed = json.loads(original)
    mixed["histories"] = {}
    config_path.write_text(json.dumps(mixed))
    seal_run(daily_root, "daily-status.json")
    with pytest.raises(ValueError, match="feature preparation differs"):
        export(daily_root, tmp_path / "mixed-config-export", outputs=outputs)
    config_path.write_bytes(original)
    seal_run(daily_root, "daily-status.json")
    (daily_root / "training/feature-state.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        export(daily_root, tmp_path / "bad", outputs=outputs)


class S3:
    def __init__(self):
        self.objects = {}
        self.serial = 0

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, tag = self.objects[Bucket, Key]
        return {"Body": io.BytesIO(body), "ETag": tag}

    def put_object(self, Bucket, Key, Body, **kwargs):
        previous = self.objects.get((Bucket, Key))
        if (kwargs.get("IfNoneMatch") == "*" and previous) or (
            "IfMatch" in kwargs and (not previous or previous[1] != kwargs["IfMatch"])
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.serial += 1
        tag = str(self.serial)
        self.objects[Bucket, Key] = (Body.read() if hasattr(Body, "read") else Body, tag)
        return {"ETag": tag}


def test_release_claim_is_conditional_and_does_not_expire():
    s3 = S3()
    a = ReleaseState(s3, "s3://bucket/state", "key")
    b = ReleaseState(s3, "s3://bucket/state", "key")
    a.claim("operation-a", "2026-10-07")
    with pytest.raises(ClientError):
        b.claim("operation-b", "2026-10-07")
    c = ReleaseState(s3, "s3://bucket/state", "key")
    with pytest.raises(ValueError, match="active release"):
        c.claim("operation-b", "2026-10-08")


def test_immutable_package_retry(tmp_path):
    from recommend_tf.train.comm.artifact_integrity import files

    (tmp_path / "weights").write_text("synthetic")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "version": "doku-tf-inference/v1",
                "model_version": "fixture",
                "files": files(tmp_path),
            }
        )
    )
    s3 = S3()
    assert publish_package(s3, tmp_path, "s3://bucket/models", "key") == publish_package(
        s3, tmp_path, "s3://bucket/models", "key"
    )
    (tmp_path / "weights").write_text("corrupt")
    with pytest.raises(ValueError):
        publish_package(s3, tmp_path, "s3://bucket/models", "key")


class SageMaker:
    def __init__(self):
        self.models, self.configs = {}, {}
        self.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "old"}
        self.updates = []

    def describe_model(self, ModelName):
        return self.models.get(ModelName)

    def create_model(self, **kwargs):
        self.models[kwargs["ModelName"]] = kwargs

    def describe_endpoint_config(self, EndpointConfigName):
        return self.configs.get(EndpointConfigName)

    def create_endpoint_config(self, **kwargs):
        self.configs[kwargs["EndpointConfigName"]] = kwargs

    def describe_endpoint(self, **kwargs):
        return self.endpoint

    def update_endpoint(self, **kwargs):
        self.updates.append(kwargs)
        self.endpoint = {
            "EndpointStatus": "Updating",
            "EndpointConfigName": "old",
            "PendingDeploymentSummary": {"EndpointConfigName": kwargs["EndpointConfigName"]},
        }


def test_endpoint_waits_for_actual_configuration_and_probe():
    state = ReleaseState(S3(), "s3://bucket/state", "key")
    active = state.claim("op", "2026-10-07")
    active.update(stage="uploaded", model_version="fixture", model_uri="s3://bucket/model.tar.gz")
    state.save()
    sm = SageMaker()
    binding = {
        "endpoint_name": "fixture",
        "image_uri": "registry/image@sha256:abc",
        "execution_role_arn": "role",
        "artifact_contract": "doku-tf-inference/v1",
        "instance_count": 1,
        "instance_type": "ml.c5.xlarge",
        "alarm_names": ["health"],
        "wait_seconds": 60,
        "termination_wait_seconds": 60,
        "timeout_seconds": 3600,
    }
    seen = []

    def probe(*args):
        seen.append(args)

    assert deploy_step(sm, None, state, binding, probe=probe) == "pending"
    assert deploy_step(sm, None, state, binding, probe=probe) == "pending"
    assert len(sm.updates) == 1 and not seen
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": active["endpoint_config"]}
    assert deploy_step(sm, None, state, binding, probe=probe) == "deployed"
    assert seen


def deployment_fixture():
    state = ReleaseState(S3(), "s3://bucket/state", "key")
    active = state.claim("op", "2026-10-07")
    active.update(stage="uploaded", model_version="fixture", model_uri="s3://bucket/model.tar.gz")
    state.save()
    binding = {
        "endpoint_name": "fixture",
        "image_uri": "registry/image@sha256:abc",
        "execution_role_arn": "role",
        "artifact_contract": "doku-tf-inference/v1",
        "instance_count": 1,
        "instance_type": "ml.c5.xlarge",
        "alarm_names": ["health"],
        "wait_seconds": 60,
        "termination_wait_seconds": 60,
        "timeout_seconds": 3600,
    }
    return state, SageMaker(), binding


def test_probe_failure_restores_previous_configuration():
    from recommend_tf.platforms.sagemaker.publish_model import rollback_step

    state, sm, binding = deployment_fixture()
    active = state.value["active"]

    def fail(*args):
        raise ValueError("incorrect predictions")

    deploy_step(sm, None, state, binding, probe=fail)
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": active["endpoint_config"]}
    assert deploy_step(sm, None, state, binding, probe=fail) == "pending"
    assert active["stage"] == "rollback_pending"
    assert sm.updates[-1]["EndpointConfigName"] == "old"
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "old"}
    assert rollback_step(sm, state, binding) == "rolled_back"


def test_uncertain_submission_never_blindly_repeats():
    state, sm, binding = deployment_fixture()

    def uncertain(**kwargs):
        raise TimeoutError("lost response")

    sm.update_endpoint = uncertain
    with pytest.raises(TimeoutError):
        deploy_step(sm, None, state, binding, probe=lambda *args: None)
    with pytest.raises(ValueError, match="ambiguous prior"):
        deploy_step(sm, None, state, binding, probe=lambda *args: None)


def test_executor_takeover_requires_stopped_task():
    from recommend_tf.platforms.sagemaker.daily_pipeline import acquire_executor

    state, _, _ = deployment_fixture()

    class ECS:
        def __init__(self):
            self.status = "RUNNING"

        def describe_tasks(self, **kwargs):
            return {"tasks": [{"lastStatus": self.status}]}

    ecs = ECS()
    acquire_executor(state, ecs, "cluster", "old")
    with pytest.raises(ValueError, match="not confirmed stopped"):
        acquire_executor(state, ecs, "cluster", "new")
    ecs.status = "STOPPED"
    acquire_executor(state, ecs, "cluster", "new")
    assert state.value["active"]["executor"] == "new"


def test_explicit_rollback_keeps_high_water():
    from recommend_tf.platforms.sagemaker.publish_model import request_rollback, rollback_step

    state, sm, binding = deployment_fixture()
    old = dict(
        state.value["active"],
        stage="deployed",
        endpoint_config="old",
        cutoff="2026-10-06",
        model_version="older",
    )
    state.value["history"].append(old)
    current = state.value["active"]
    current.update(stage="deployed", endpoint_config="new")
    state.finish(current)
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "new"}
    request_rollback(state, sm, "fixture", "older", "rollback-op")
    assert rollback_step(sm, state, binding) == "pending"
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "old"}
    assert rollback_step(sm, state, binding) == "rolled_back"
    state.finish(state.value["active"])
    assert state.value["high_water"] == "2026-10-07"
    assert state.value["online"]["model_version"] == "older"
    with pytest.raises(ValueError, match="equal/older"):
        state.claim("stale", "2026-10-06")


def test_training_submission_reconciles_instead_of_duplicating(tmp_path):
    from recommend_tf.platforms.sagemaker.daily_pipeline import step

    state, sm, _ = deployment_fixture()
    request = {
        "TrainingJobName": "fixture-job",
        "Environment": {"config": "fixed"},
        "AlgorithmSpecification": {"TrainingImage": "image"},
    }
    state.value["active"].update(stage="training", training_request=request)
    calls = []
    sm.describe_training_job = lambda **kwargs: dict(request, TrainingJobStatus="InProgress")
    sm.create_training_job = lambda **kwargs: calls.append(kwargs)
    assert step(None, sm, None, state, {}, {}, {}, work=tmp_path) == "pending"
    assert not calls
    sm.describe_training_job = lambda **kwargs: dict(
        request, Environment={"config": "changed"}, TrainingJobStatus="InProgress"
    )
    with pytest.raises(ValueError, match="identity differs"):
        step(None, sm, None, state, {}, {}, {}, work=tmp_path)


def test_ready_contract_tampering_and_maturity_fail_closed(tmp_path):
    config, ready = fixture(tmp_path)
    bad = copy.deepcopy(ready)
    bad["sources"][0]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        freeze(config, bad, tmp_path / "bad-digest")
    bad = copy.deepcopy(ready)
    bad["sources"][0].pop("maturity_evidence")
    with pytest.raises(ValueError, match="maturity"):
        freeze(config, bad, tmp_path / "bad-maturity")


def test_run_download_rejects_path_escape(tmp_path):
    from recommend_tf.platforms.sagemaker.daily_pipeline import fetch_run

    s3 = S3()
    s3.put_object(
        Bucket="bucket",
        Key="run/COMMITTED.json",
        Body=json.dumps({"version": "doku-tf-run/v1", "files": {"../outside": "hash"}}).encode(),
    )
    with pytest.raises(ValueError, match="invalid artifact path"):
        fetch_run(s3, "s3://bucket/run", tmp_path / "run", 1000)


def test_prepare_job_freezes_input_and_is_content_idempotent(tmp_path):
    from recommend_tf.platforms.sagemaker.daily_pipeline import prepare_job

    config, ready = fixture(tmp_path)
    binding = {
        "work_prefix": "s3://bucket/work",
        "kms_key": "key",
        "training_request": {
            "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.c5.xlarge"},
            "AlgorithmSpecification": {"TrainingImage": "registry/train@sha256:abc"},
            "Environment": {"DOKU_SOURCE_SHA256": "source", "DOKU_LOCK_SHA256": "lock"},
        },
    }
    s3 = S3()
    first, _ = prepare_job(s3, config, ready, binding, "op", tmp_path / "first")
    second, _ = prepare_job(s3, config, ready, binding, "op", tmp_path / "second")
    assert first == second
    cfg = json.loads(s3.objects["bucket", "work/op/daily/config.json"][0])
    record = json.loads(s3.objects["bucket", "work/op/daily/input/record.json"][0])
    assert cfg["dataset_config"]["dataset"]["manifest_uri"] == record["manifest_uri"]
    assert first["Environment"]["DOKU_MODE"] == "daily"
    assert (
        first["Environment"]["DOKU_CONFIG_SHA256"]
        == hashlib.sha256(s3.objects["bucket", "work/op/daily/config.json"][0]).hexdigest()
    )


def test_failure_run_cannot_export(tmp_path):
    from recommend_tf.train.comm.artifact_integrity import seal_run

    root = tmp_path / "run"
    root.mkdir()
    for name in ("experiment-config.json", "resolved-config.json"):
        (root / name).write_text("{}")
    (root / "daily-status.json").write_text('{"status": "failed"}')
    seal_run(root, "daily-status.json")
    with pytest.raises(ValueError, match="successful selected"):
        export(root, tmp_path / "out", outputs={})


def test_eventually_consistent_old_endpoint_is_not_false_rollback():
    state, sm, binding = deployment_fixture()
    deploy_step(sm, None, state, binding, probe=lambda *args: None)
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "old"}
    assert deploy_step(sm, None, state, binding, probe=lambda *args: None) == "pending"
    assert state.value["active"]["stage"] == "deploying"


def test_export_destination_reservation_is_create_only(tmp_path):
    from recommend_tf.train.comm.artifact_integrity import reserve_output

    destination = tmp_path / "package"
    with reserve_output(destination):
        with pytest.raises(FileExistsError):
            with reserve_output(destination):
                pass
    assert not destination.exists()


def test_resume_budget_change_keeps_operation_identity():
    from recommend_tf.platforms.sagemaker.daily_pipeline import operation_identity

    binding = {
        "poll_seconds": 10,
        "max_wait_seconds": 60,
        "download_limit_bytes": 100,
        "export_timeout_seconds": 30,
        "deployment": {"endpoint_name": "one"},
    }
    first = operation_identity({}, {"cutoff": "2026-10-07"}, binding)
    larger = dict(binding, max_wait_seconds=600, download_limit_bytes=1000)
    assert operation_identity({}, {"cutoff": "2026-10-07"}, larger) == first
    larger["deployment"] = {"endpoint_name": "two"}
    assert operation_identity({}, {"cutoff": "2026-10-07"}, larger) != first


@pytest.mark.parametrize("entry", ["experiment", "daily"])
def test_output_race_preserves_other_run(tmp_path, monkeypatch, entry):
    import importlib

    module = importlib.import_module(
        "recommend_tf.train.experiment"
        if entry == "experiment"
        else "recommend_tf.train.daily_train_model"
    )
    config, ready = fixture(tmp_path)
    frozen = tmp_path / "frozen"
    resolved = freeze(config, ready, frozen)
    root = tmp_path / "raced"

    def competing_train(*args, **kwargs):
        root.mkdir()
        (root / "COMMITTED.json").write_text("other successful run")
        raise FileExistsError(root)

    monkeypatch.setattr(module, "train", competing_train)
    with pytest.raises(FileExistsError):
        if entry == "experiment":
            module.execute(config, root)
        else:
            module.execute(resolved, root, cutoff=ready["cutoff"], frozen_input=frozen / "input")
    assert (root / "COMMITTED.json").read_text() == "other successful run"
    assert not list(root.glob("*-status.json"))


def test_endpoint_drift_between_intent_and_update_is_not_overwritten():
    state, sm, binding = deployment_fixture()
    calls = 0

    def changed(**kwargs):
        nonlocal calls
        calls += 1
        return {
            "EndpointStatus": "InService",
            "EndpointConfigName": "old" if calls == 1 else "external-new",
        }

    sm.describe_endpoint = changed
    with pytest.raises(ValueError, match="changed outside"):
        deploy_step(sm, None, state, binding, probe=lambda *args: None)
    assert not sm.updates


def test_external_endpoint_change_during_rollback_is_not_overwritten():
    from recommend_tf.platforms.sagemaker.publish_model import rollback_step

    state, sm, binding = deployment_fixture()
    state.value["active"].update(
        stage="rollback_pending", previous_config="old", endpoint_config="ours"
    )
    sm.endpoint = {"EndpointStatus": "InService", "EndpointConfigName": "external-new"}
    with pytest.raises(ValueError, match="changed outside"):
        rollback_step(sm, state, binding)
    assert not sm.updates


def test_failed_platform_rollback_attempts_restore_without_unlocking():
    state, sm, binding = deployment_fixture()
    deploy_step(sm, None, state, binding, probe=lambda *args: None)
    active = state.value["active"]
    sm.endpoint = {
        "EndpointStatus": "UpdateRollbackFailed",
        "EndpointConfigName": active["endpoint_config"],
    }
    assert deploy_step(sm, None, state, binding, probe=lambda *args: None) == "pending"
    assert state.value["active"]["stage"] == "rollback_pending"
    assert sm.updates[-1]["EndpointConfigName"] == "old"


def test_artifact_metadata_does_not_collide_at_same_logical_model_version(tmp_path):
    from recommend_tf.train.comm.artifact_integrity import files

    s3 = S3()
    uris = []
    for index in (1, 2):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "lineage.json").write_text(json.dumps({"run": index}))
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "version": "doku-tf-inference/v1",
                    "model_version": "same-weights",
                    "files": files(folder),
                }
            )
        )
        uris.append(publish_package(s3, folder, "s3://bucket/models", "key"))
    assert uris[0] != uris[1]


def test_memory_budget_start_failure_is_cleaned_up(tmp_path):
    from recommend_tf.train.comm import runtime_budget
    from recommend_tf.train.train_rerank import train

    config, _ = fixture(tmp_path)
    config["dataset_config"]["execution"]["memory_limit_bytes"] = 1
    with pytest.raises(MemoryError):
        train(config, tmp_path / "run")
    assert runtime_budget._active is None


def test_ready_preflight_failure_does_not_claim_endpoint(tmp_path):
    from recommend_tf.platforms.sagemaker.daily_pipeline import preflight_ready

    config, ready = fixture(tmp_path)
    ready["sources"].pop()
    s3 = S3()
    state = ReleaseState(s3, "s3://bucket/state", "key")
    with pytest.raises(ValueError, match="contiguous"):
        preflight_ready(state, config, ready, "new")
    assert state.value["active"] is None and not s3.objects


def test_completed_and_resuming_operations_do_not_reread_upstream(monkeypatch):
    from recommend_tf.platforms.sagemaker import daily_pipeline

    state, _, _ = deployment_fixture()

    def forbidden(*args):
        raise AssertionError("upstream must not be reread")

    monkeypatch.setattr(daily_pipeline, "freeze", forbidden)
    daily_pipeline.preflight_ready(state, {}, {}, "op")  # uploaded stage
    state.value["history"] = [dict(state.value["active"], stage="deployed")]
    state.value["active"] = None
    daily_pipeline.preflight_ready(state, {}, {}, "op")


def test_input_failure_after_claim_can_release_slot(tmp_path, monkeypatch):
    from recommend_tf.platforms.sagemaker import daily_pipeline

    state = ReleaseState(S3(), "s3://bucket/state", "key")
    state.claim("op", "2026-10-07")

    def fail(*args):
        raise ValueError("corrupt source after preflight")

    monkeypatch.setattr(daily_pipeline, "prepare_job", fail)
    assert daily_pipeline.step(None, None, None, state, {}, {}, {}, work=tmp_path) == "input_failed"
    state.finish(state.value["active"])
    assert state.value["active"] is None
    assert state.claim("next", "2026-10-08")["stage"] == "claimed"


@pytest.mark.parametrize("code", ["AccessDeniedException", "ValidationException"])
def test_explicit_endpoint_rejection_can_resume(code):
    state, sm, binding = deployment_fixture()
    original = sm.update_endpoint

    def rejected(**kwargs):
        raise ClientError({"Error": {"Code": code, "Message": "rejected"}}, "UpdateEndpoint")

    sm.update_endpoint = rejected
    with pytest.raises(ClientError):
        deploy_step(sm, None, state, binding, probe=lambda *args: None)
    assert state.value["active"]["attempted"] is False
    sm.update_endpoint = original
    assert deploy_step(sm, None, state, binding, probe=lambda *args: None) == "pending"
    assert len(sm.updates) == 1


def test_training_job_absence_validation_response():
    from recommend_tf.platforms.sagemaker.publish_model import describe

    class Client:
        message = "Requested resource not found."

        def describe_training_job(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": self.message}},
                "DescribeTrainingJob",
            )

    client = Client()
    assert describe(client, "describe_training_job", TrainingJobName="fixture") is None
    client.message = "Invalid training job request"
    with pytest.raises(ClientError):
        describe(client, "describe_training_job", TrainingJobName="fixture")


def test_monitoring_callback_failure_preserves_daily_success(tmp_path):
    config, ready = fixture(tmp_path)
    resolved = freeze(config, ready, tmp_path / "frozen")

    def fail_monitoring(root):
        raise RuntimeError("synthetic monitoring failure")

    status = daily(
        resolved,
        tmp_path / "run",
        cutoff=ready["cutoff"],
        frozen_input=tmp_path / "frozen/input",
        on_completed=fail_monitoring,
    )
    assert status["status"] == "completed"
    assert status["monitoring"] == "failed"
    from recommend_tf.train.comm.artifact_integrity import verify_run

    record = verify_run(tmp_path / "run")
    assert "training/selected-model-metrics.json" in record["files"]
