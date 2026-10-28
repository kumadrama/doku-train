from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from recommend.train.comm import metric_events
from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.training_pipeline import source_identity
from recommend.train.platforms.sagemaker.artifacts import verify_archive
from recommend.train.platforms.sagemaker.config import Binding, Runtime
from recommend.train.platforms.sagemaker.entrypoint import run_package
from recommend.train.platforms.sagemaker.events import EventSink
from recommend.train.platforms.sagemaker.submit import canonical, prepare, put_once, submit
from recommend.train.tests.experiment_fixture import make_experiment


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.puts = []

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        name = kwargs["Bucket"], kwargs["Key"]
        if name in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[name] = kwargs["Body"]


def fixture(tmp_path, model="deepfm"):
    config = make_experiment(tmp_path / "fixture", model)
    source = source_identity()
    binding = Binding(
        version="doku-sagemaker-binding/v1",
        environment="test",
        region="us-east-2",
        account_id="123456789012",
        role_arn="arn:aws:iam::123456789012:role/test",
        subnet_ids=["subnet-test"],
        security_group_ids=["sg-test"],
        input_prefixes=["s3://synthetic/input/"],
        experiment_uri="s3://synthetic/experiments/",
        kms_key_arn="arn:aws:kms:us-east-2:123456789012:key/test",
    )
    runtime = Runtime(
        version="doku-sagemaker-runtime/v1",
        environment="test",
        binding="fixture.json",
        image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/test@sha256:" + "a" * 64,
        source_revision=source["revision"],
        source_sha256=source["working_source_sha256"],
        lock_sha256=source["lock_sha256"],
        loss_interval_steps=1,
    )
    local = tmp_path / "fixture/dataset"
    manifest = json.loads((local / "manifest.json").read_text())
    s3 = MemoryS3()
    shards = {}
    for i, part in enumerate(manifest["partitions"]):
        original = Path(part["uri"].removeprefix("file://"))
        shards[f"s3://synthetic/input/shard-{i}.parquet"] = original
        part["uri"] = f"s3://synthetic/input/shard-{i}.parquet"
    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    s3.objects[("synthetic", "input/manifest.json")] = canonical(manifest)
    for name in ["features", "labels"]:
        s3.objects[("synthetic", f"input/contracts/{name}.json")] = (
            local / f"contracts/{name}.json"
        ).read_bytes()
    config["dataset"]["manifest_uri"] = "s3://synthetic/input/manifest.json"
    return config, runtime, binding, s3, shards


def test_prepare_does_not_write_cloud_and_freezes_input(tmp_path):
    config, runtime, binding, s3, _ = fixture(tmp_path)
    request = prepare(config, runtime, binding, "trial-1", s3, tmp_path / "prepared")
    assert not s3.puts
    assert request["ResourceConfig"] == {
        "InstanceType": "ml.m5.2xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    }
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 86400
    before = (tmp_path / "prepared/package.json").read_bytes()
    s3.objects[("synthetic", "input/manifest.json")] = b"changed"
    config["batch_size"] *= 2
    assert (tmp_path / "prepared/package.json").read_bytes() == before


def test_outside_input_is_rejected(tmp_path):
    config, runtime, binding, s3, _ = fixture(tmp_path)
    manifest = json.loads(s3.objects[("synthetic", "input/manifest.json")])
    manifest["partitions"][0]["uri"] = "s3://other/private.parquet"
    s3.objects[("synthetic", "input/manifest.json")] = canonical(manifest)
    with pytest.raises(ValueError, match="outside approved"):
        prepare(config, runtime, binding, "trial", s3, tmp_path / "prepared")
    assert not s3.puts


def test_immutable_encrypted_upload():
    s3 = MemoryS3()
    put_once(s3, "s3://synthetic/a", b"x", "key")
    put_once(s3, "s3://synthetic/a", b"x", "key")
    assert s3.puts[0]["IfNoneMatch"] == "*"
    assert s3.puts[0]["ServerSideEncryption"] == "aws:kms"
    assert s3.puts[0]["SSEKMSKeyId"] == "key"
    with pytest.raises(ValueError, match="collision"):
        put_once(s3, "s3://synthetic/a", b"y", "key")


def test_lost_create_response_reuses_job(tmp_path):
    config, runtime, binding, s3, _ = fixture(tmp_path)
    path = tmp_path / "prepared"
    request = prepare(config, runtime, binding, "trial", s3, path)

    class SM:
        created = None
        calls = 0

        def describe_training_job(self, **kwargs):
            if self.created is None:
                raise ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "not found"}},
                    "DescribeTrainingJob",
                )
            return {"Environment": request["Environment"], "TrainingJobArn": "arn:test"}

        def create_training_job(self, **kwargs):
            self.created = kwargs
            self.calls += 1
            raise TimeoutError("response lost")

    sm = SM()
    with pytest.raises(TimeoutError):
        submit(path, s3, sm)
    assert submit(path, s3, sm)["reused"]
    assert sm.calls == 1


def test_events_keep_step_and_do_not_emit_null_scalar(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    s3 = MemoryS3()
    sink = EventSink(tmp_path, s3, "s3://synthetic/events/", "key", 30)
    with metric_events.observe(sink, 2):
        metric_events.learning_rate(123, 2, 0.002)
        metric_events.training_step(1, 1, {"task": 0.9})
        metric_events.training_step(2, 1, {"task": 0.4})
        metric_events.training_step(3, 2, {"task": 0.3})
        metric_events.training_step(4, 2, {"task": 0.2})
        metric_events.emit(dict(split="validation", task="task", metric="auc", step=4, value=None))
    sink.close()
    files = list(tmp_path.rglob("events.out.tfevents.*"))
    assert len(files) == 1
    events = EventAccumulator(str(files[0])).Reload()
    assert [x.step for x in events.Scalars("train/task/loss")] == [2, 4]
    assert "validation/task/auc" not in events.Tags()["scalars"]
    assert [(x.step, x.value) for x in events.Scalars("train/learning_rate")] == [
        (2, pytest.approx(0.002))
    ]
    assert len(s3.puts) == 1


@pytest.mark.parametrize("profiling", [False, True])
def test_frozen_package_train_test_archive(tmp_path, monkeypatch, profiling):
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset

    config, runtime, binding, s3, shards = fixture(tmp_path)
    config["profiling"] = {"enabled": profiling, "print_logs": False}
    prepared = tmp_path / "prepared"
    prepare(config, runtime, binding, "trial", s3, prepared)
    payload = prepared / "package.json"
    # Reader receives frozen contracts; shards are synthetic local files only.
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))
    s3.objects[("synthetic", "input/manifest.json")] = b"original changed after submission"
    root = tmp_path / "ml"
    run_package(payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest())
    verify_archive(root / "model")
    if profiling:
        records = [
            json.loads(p.read_text()) for p in (root / "model/timing").glob("*-summary.json")
        ]
        assert {r["phase"] for r in records} == {"train", "test", "archive"}
        assert all(r["status"] == "complete" for r in records)
        assert len(list((root / "output/data/timing").glob("*-summary.json"))) == 3
        manifest = json.loads((root / "model/artifact-manifest.json").read_text())
        assert len([name for name in manifest["files"] if name.startswith("timing/")]) == 6
    else:
        assert not (root / "model/timing").exists()
    assert not list((root / "model").rglob("*.npy"))
    assert (root / "model/test-metrics.json").is_file()
    metrics = json.loads((root / "model/metrics.json").read_text())
    steps = [e["global_step"] for e in metrics["history"]]
    assert steps == sorted(set(steps))
    assert metrics["selection"]["global_step"] in steps
    (root / "model/checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(root / "model")


@pytest.mark.parametrize("failure_stage", ["test_run", "archive_run"])
def test_failed_evaluation_or_archive_never_marks_complete(tmp_path, monkeypatch, failure_stage):
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset
    from recommend.train.platforms.sagemaker import entrypoint

    config, runtime, binding, s3, shards = fixture(tmp_path)
    prepared = tmp_path / "prepared"
    prepare(config, runtime, binding, "trial", s3, prepared)
    payload = prepared / "package.json"
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic stage failure")

    monkeypatch.setattr(entrypoint, failure_stage, fail)
    root = tmp_path / "ml"
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        run_package(
            payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest()
        )
    assert not (root / "model/artifact-manifest.json").exists()
    assert (root / "output/data/resources.json").is_file()


def test_submission_metadata_can_reference_approved_data_only(tmp_path):
    config, runtime, binding, s3, _ = fixture(tmp_path)
    for name in ["manifest.json", "contracts/features.json", "contracts/labels.json"]:
        s3.objects[("synthetic", "experiments/submissions/trial/dataset/" + name)] = s3.objects[
            ("synthetic", "input/" + name)
        ]
    config["dataset"]["manifest_uri"] = (
        "s3://synthetic/experiments/submissions/trial/dataset/manifest.json"
    )
    prepare(config, runtime, binding, "trial", s3, tmp_path / "prepared")
    manifest = json.loads(
        s3.objects[("synthetic", "experiments/submissions/trial/dataset/manifest.json")]
    )
    manifest["partitions"][0]["uri"] = (
        "s3://synthetic/experiments/submissions/trial/private.parquet"
    )
    s3.objects[("synthetic", "experiments/submissions/trial/dataset/manifest.json")] = canonical(
        manifest
    )
    with pytest.raises(ValueError, match="outside approved"):
        prepare(config, runtime, binding, "bad", s3, tmp_path / "bad")
    assert not s3.puts


@pytest.mark.parametrize("stages", [["train"], ["train", "test"]])
def test_tree_package_stages_and_archive(tmp_path, monkeypatch, stages):
    import os
    import subprocess
    import sys

    if os.environ.get("DOKU_TREE_TEST_CHILD") != "1":
        node = "stages0" if stages == ["train"] else "stages1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{__file__}::test_tree_package_stages_and_archive[{node}]",
                "-q",
            ],
            env={**os.environ, "DOKU_TREE_TEST_CHILD": "1"},
            check=True,
            timeout=120,
        )
        return
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset

    config, runtime, binding, s3, shards = fixture(tmp_path, "xgboost")
    config["execution"]["intra_op_threads"] = 1
    runtime = runtime.model_copy(update={"stages": stages})
    prepared = tmp_path / "prepared"
    prepare(config, runtime, binding, "tree-trial", s3, prepared)
    payload = prepared / "package.json"
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))
    root = tmp_path / "ml"
    run_package(payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest())
    verify_archive(root / "model")
    assert (root / "model/checkpoint.xgb.zip").is_file()
    assert not (root / "model/checkpoint.pt").exists()
    assert (root / "model/test-metrics.json").exists() == ("test" in stages)
    assert (
        not (root / "work/runs/tree-trial/encoded-test").exists() if "test" not in stages else True
    )


def test_tree_yaml_is_frozen_in_submission_and_run(tmp_path, monkeypatch):
    import shutil

    from recommend.train import train_rerank
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset
    from recommend.train.tests.tree_binning_fixture import configure_tree_binning

    config, runtime, binding, s3, shards = fixture(tmp_path)
    teacher_path = tmp_path / "teacher.yaml"
    configure_tree_binning(config, teacher_path)
    prepare(config, runtime, binding, "tree-bins", s3, tmp_path / "prepared")
    payload = tmp_path / "prepared/package.json"
    package = json.loads(payload.read_text())
    assert package["tree_config"]["model_type"] == "rerank_tree"
    teacher_path.unlink()  # Container must not access the submitter's YAML.
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    for uri, path in shards.items():
        shutil.copyfile(path, mirror / uri.rsplit("/", 1)[1])
    original = train_rerank.execute_train

    def local_teacher(teacher, **kwargs):
        kwargs["local_root"] = mirror
        return original(teacher, **kwargs)

    monkeypatch.setattr(train_rerank, "execute_train", local_teacher)
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))
    s3.objects[("synthetic", "input/manifest.json")] = b"changed after prepare"
    root = tmp_path / "ml"
    run_package(payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest())
    verify_archive(root / "model")
    artifact = json.loads((root / "model/tree-buckets.json").read_text())
    assert artifact["source"]["config"]["model_params"] == package["tree_config"]["model_params"]
    state = json.loads((root / "model/fitted-feature-state/state.json").read_text())
    assert state["tree_buckets"] == artifact


def test_submit_cli_resolves_teacher_relative_to_yaml(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import yaml

    from recommend.train.comm.tree_pretraining import load_tree_config
    from recommend.train.platforms.sagemaker import __main__ as cli
    from recommend.train.tests.tree_binning_fixture import make_tree_experiment

    directory = tmp_path / "configs"
    directory.mkdir()
    config = make_tree_experiment(directory)
    config["tree_binning"]["config_path"] = "teacher.yaml"
    path = directory / "nn.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    binding = SimpleNamespace(region="us-east-2", account_id="123456789012")
    monkeypatch.setattr(cli, "load_runtime", lambda _: (object(), binding))
    client = SimpleNamespace(get_caller_identity=lambda: {"Account": binding.account_id})
    monkeypatch.setattr(cli.boto3, "Session", lambda **_: SimpleNamespace(client=lambda _: client))
    captured = []

    def prepare_config(value, *args, **kwargs):
        assert value["tree_binning"]["config_path"] == str(directory / "teacher.yaml")
        captured.append(load_tree_config(value))
        return {"TrainingJobName": "synthetic"}

    monkeypatch.setattr(cli, "prepare", prepare_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sagemaker",
            "--runtime-config",
            "runtime.yaml",
            "--profile",
            "test",
            "submit",
            "--config",
            str(path),
            "--run-id",
            "synthetic",
            "--prepared",
            str(tmp_path / "prepared"),
            "--dry-run",
        ],
    )
    cli.main()
    assert captured[0]["model_type"] == "rerank_tree"


def test_failed_deepfm_exports_diagnostic_timing(tmp_path, monkeypatch):
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset
    from recommend.train.models.rerank.runner import NeuralRunner

    config, runtime, binding, s3, shards = fixture(tmp_path)
    config["profiling"] = {"enabled": True}
    prepared = tmp_path / "prepared"
    prepare(config, runtime, binding, "trial", s3, prepared)
    payload = prepared / "package.json"
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))

    def fail(self, arrays):
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(NeuralRunner, "evaluate", fail)
    root = tmp_path / "ml"
    with pytest.raises(RuntimeError, match="synthetic model"):
        run_package(
            payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest()
        )
    reports = list((root / "output/data/timing").glob("*-summary.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == "failed"
    assert not (root / "model/artifact-manifest.json").exists()


def test_event_upload_failure_still_exports_timings(tmp_path, monkeypatch):
    from recommend.train.comm.datasvr.published_dataset import PublishedDataset

    config, runtime, binding, s3, shards = fixture(tmp_path)
    config["profiling"] = {"enabled": True}
    prepared = tmp_path / "prepared"
    prepare(config, runtime, binding, "trial", s3, prepared)
    payload = prepared / "package.json"
    monkeypatch.setattr(PublishedDataset, "_local_path", lambda self, uri: shards.get(uri))
    original = EventSink.close

    def fail_close(self):
        original(self)
        raise RuntimeError("synthetic event upload failure")

    monkeypatch.setattr(EventSink, "close", fail_close)
    root = tmp_path / "ml"
    with pytest.raises(RuntimeError, match="event upload"):
        run_package(
            payload, root, s3, expected_sha=hashlib.sha256(payload.read_bytes()).hexdigest()
        )
    assert len(list((root / "output/data/timing").glob("*-summary.json"))) == 2
    assert not (root / "model/artifact-manifest.json").exists()


@pytest.mark.parametrize("missing", ["train", "test", "archive", "jsonl"])
def test_timing_archive_requires_complete_invocations(tmp_path, missing):
    from recommend.train.comm.timing import session
    from recommend.train.platforms.sagemaker.artifacts import validate_timing_files

    config = {"profiling": {"enabled": True}}
    package = {"config": config, "runtime": {"stages": ["train", "test"]}}
    for phase in ["train", "test", "archive"]:
        with session(config, tmp_path, phase):
            pass
    names = {str(p.relative_to(tmp_path)) for p in (tmp_path / "timing").iterdir()}
    validate_timing_files(tmp_path, package, names, archived=True)
    if missing == "jsonl":
        names = {
            name
            for name in names
            if not name.startswith("timing/test-") or not name.endswith(".jsonl")
        }
    else:
        names = {name for name in names if not name.startswith(f"timing/{missing}-")}
    with pytest.raises(ValueError, match="timing"):
        validate_timing_files(tmp_path, package, names, archived=True)


@pytest.mark.parametrize("instance", ["ml.m5.2xlarge", "ml.m5.4xlarge"])
def test_supported_instance_survives_package_validation(tmp_path, instance):
    config, runtime, binding, s3, _ = fixture(tmp_path)
    runtime = Runtime.model_validate({**runtime.model_dump(), "instance_type": instance})
    request = prepare(config, runtime, binding, "instance-test", s3, tmp_path / "prepared")
    package = json.loads((tmp_path / "prepared/package.json").read_text())
    assert Runtime.model_validate(package["runtime"]).instance_type == instance
    assert request["ResourceConfig"]["InstanceType"] == instance
    assert request["ResourceConfig"]["InstanceCount"] == 1


@pytest.mark.parametrize(
    "instance,count", [("ml.m5.8xlarge", 1), ("ml.p3.2xlarge", 1), ("ml.m5.4xlarge", 2)]
)
def test_other_instances_and_multi_node_rejected(tmp_path, instance, count):
    _, runtime, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="only single"):
        Runtime.model_validate(
            {**runtime.model_dump(), "instance_type": instance, "instance_count": count}
        )
