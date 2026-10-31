import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recommend_tf.platforms.sagemaker import audit_run, executor
from recommend_tf.train.tests.test_daily_publication import S3


def test_executor_digest_environment_and_identity(tmp_path):
    s3 = S3()
    values = {
        "config": {"model": "fixture"},
        "ready": {"cutoff": "2026-10-07"},
        "binding": {
            "environment": "test",
            "region": "us-east-1",
            "ecs_cluster": "test",
            "state_uri": "s3://bucket/state",
            "work_prefix": "s3://bucket/work",
            "model_prefix": "s3://bucket/models",
            "kms_key": "key",
            "poll_seconds": 1,
            "max_wait_seconds": 2,
            "download_limit_bytes": 100,
            "export_timeout_seconds": 2,
            "deployment": {"instance_count": 1, "instance_type": "ml.m5.large"},
            "outputs": {"a": "unit"},
            "probe": {"tasks": ["a"], "fixture_kind": "synthetic"},
        },
    }
    args = SimpleNamespace(
        region="us-east-1",
        bootstrap_run=None,
        bootstrap_evidence_uri=None,
        rollback_version=None,
        retry_id="",
    )
    for name, value in values.items():
        body = json.dumps(value).encode()
        s3.put_object(Bucket="bucket", Key=name, Body=body)
        setattr(args, name + "_uri", "s3://bucket/" + name)
        setattr(args, name + "_sha256", hashlib.sha256(body).hexdigest())
    seen = []

    def invoke(options):
        assert json.loads(Path(options.config).read_text()) == values["config"]
        seen.append(options.executor_arn)

    def identity(_):
        return "arn:aws:ecs:us-east-1:123:task/test/id", "test"

    executor.run(args, s3=s3, metadata_uri="unused", invoke=invoke, identity=identity)
    assert len(seen) == 1
    args.config_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        executor.run(args, s3=s3, metadata_uri="unused", invoke=invoke, identity=identity)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/v4/id",
        "http://127.0.0.1/v4/id",
        "http://169.254.170.2/v4/id?redirect=x",
    ],
)
def test_metadata_refuses_non_ecs_uri(uri):
    with pytest.raises(ValueError):
        executor.task_identity(uri)


class AuditS3:
    def __init__(self, bodies):
        self.bodies = bodies

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        yield {
            "Contents": [
                {"Key": "run/" + k, "ETag": hashlib.sha256(v).hexdigest()}
                for k, v in self.bodies.items()
            ]
        }

    def get_object(self, **kwargs):
        body = self.bodies[kwargs["Key"][4:]]
        return {"Body": io.BytesIO(body), "ETag": hashlib.sha256(body).hexdigest()}

    def head_object(self, **kwargs):
        return self.get_object(**kwargs)


def audit_fixture(tmp_path):
    config = {"dataset_config": {"split": {"validation_to": "2026-10-07"}}}
    body = json.dumps(config).encode()
    source = tmp_path / "original.json"
    source.write_bytes(body)
    job = {
        "TrainingJobName": "job",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn",
        "Environment": {
            "DOKU_ARTIFACT_URI": "s3://bucket/run",
            "DOKU_CONFIG_SHA256": hashlib.sha256(body).hexdigest(),
            "DOKU_SOURCE_SHA256": "source-with-tests",
            "DOKU_LOCK_SHA256": "lock",
        },
    }
    bodies = {
        "experiment-config.json": body,
        "resolved-config.json": b"{}",
        "experiment-status.json": b'{"status":"completed","selected_epoch":21}',
        "training/selection.json": b'{"epoch":21}',
        "training/lineage.json": b'{"lock_digest":"lock","source_digest":"source"}',
    }
    return job, bodies, source


def test_audit_requires_job_config_and_parity_before_evidence(tmp_path):
    job, bodies, source = audit_fixture(tmp_path)
    sm = SimpleNamespace(describe_training_job=lambda **_: job)
    options = dict(
        job_name="job", run_uri="s3://bucket/run", source_config=source, outputs={}, limit=100000
    )

    def fail(*args, **kwargs):
        raise ValueError("parity failure")

    with pytest.raises(ValueError, match="parity"):
        audit_run.audit(AuditS3(bodies), sm, output=tmp_path / "failed", exporter=fail, **options)
    assert not (tmp_path / "failed/evidence.json").exists()
    job["TrainingJobStatus"] = "Failed"
    with pytest.raises(ValueError, match="provenance"):
        audit_run.audit(AuditS3(bodies), sm, output=tmp_path / "badjob", **options)
    job["TrainingJobStatus"] = "Completed"
    report = audit_run.audit(
        AuditS3(bodies),
        sm,
        output=tmp_path / "ok",
        exporter=lambda *a, **k: {"model_version": "v1"},
        **options,
    )
    assert report["selected_epoch"] == 21
    assert (tmp_path / "ok/evidence.json").exists()
    broken = copy.deepcopy(bodies)
    broken["training/../outside"] = b"bad"
    with pytest.raises(ValueError, match="path"):
        audit_run.audit(AuditS3(broken), sm, output=tmp_path / "traversal", **options)


def test_clean_inference_context_has_no_training_or_tests(tmp_path):
    from recommend_tf.platforms.sagemaker.build_context import prepare

    root = prepare(tmp_path / "context", kind="inference")
    assert (root / "recommend_tf/inference/app.py").exists()
    assert not (root / "recommend_tf/train").exists()
    assert not (root / "recommend_tf/inference/tests").exists()
    with pytest.raises(FileExistsError):
        prepare(root, kind="inference")


def test_training_context_keeps_schema_and_exact_config(tmp_path, monkeypatch):
    from recommend_tf.platforms.sagemaker import build_context

    config = tmp_path / "config.json"
    config.write_bytes(b'{"synthetic":true}\n')
    calls = []
    monkeypatch.setattr(build_context.subprocess, "run", lambda args, **kwargs: calls.append(args))
    root = build_context.prepare(tmp_path / "context", kind="train", config=config)
    assert (root / "config.json").read_bytes() == config.read_bytes()
    assert (root / "recommend_tf/train/models/rerank/config.schema.yml").exists()
    assert "--frozen" in calls[0] and "--no-dev" in calls[0]
