"""Container boundary tests use local files and AWS Stubber, never AWS."""

import hashlib
import json
import sys

import boto3
import pytest
import yaml
from botocore.stub import Stubber

from recommend.train import scheduled
from recommend.train.comm import training_pipeline
from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.publication import Registry
from recommend.train.tests.synthetic_config import synthetic_config


def test_kms_headers_preserve_conditional_writes(monkeypatch):
    monkeypatch.setenv("DOKU_TRAIN_S3_KMS_KEY_ARN", "arn:aws:kms:us-east-2:123456789012:key/test")
    store = ObjectStore("s3://test-bucket/models")
    store._client = boto3.client(
        "s3", region_name="us-east-2", aws_access_key_id="test", aws_secret_access_key="test"
    )
    with Stubber(store.client) as stub:
        for condition in ({"IfNoneMatch": "*"}, {"IfMatch": "etag"}):
            stub.add_response(
                "put_object",
                {},
                {
                    "Bucket": "test-bucket",
                    "Key": "models/key",
                    "Body": b"data",
                    "ServerSideEncryption": "aws:kms",
                    "SSEKMSKeyId": "arn:aws:kms:us-east-2:123456789012:key/test",
                    **condition,
                },
            )
        store.put("key", b"data")
        store.put("key", b"data", expected="etag")
        stub.assert_no_pending_responses()


def test_cleanup_keeps_consumer_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOKU_TRAIN_ARTIFACT_ROOT", tmp_path.as_uri())
    monkeypatch.setenv("DOKU_TRAIN_MODEL_NAME", "test")
    monkeypatch.setenv("DOKU_TRAIN_S3_KMS_KEY_ARN", "test")
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    registry.start("owner", "input")
    registry.finish("owner", "rejected")
    monkeypatch.setattr(sys, "argv", ["scheduled", "cleanup"])
    scheduled.main()
    result = json.loads(capsys.readouterr().out)
    assert result["completed"] and result["result"]["dry_run"]


def test_pinned_config_rejects_digest_before_parsing(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("invalid")
    with pytest.raises(ValueError, match="digest"):
        scheduled.read_config(path.as_uri(), "0" * 64, "test")


def test_container_source_identity_without_git(tmp_path, monkeypatch):
    # ROOT is recommend/train; the image carries source and lock next to recommend.
    root = tmp_path / "recommend" / "train"
    root.mkdir(parents=True)
    (root / "source.py").write_text("x = 1")
    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "build-revision.txt").write_text("a" * 40)
    monkeypatch.setattr(training_pipeline, "ROOT", root)
    value = training_pipeline.source_identity()
    assert value["revision"] == "a" * 40
    assert value["lock_sha256"] == hashlib.sha256(b"lock").hexdigest()
    (tmp_path / "build-revision.txt").write_text("invalid")
    with pytest.raises(ValueError, match="revision"):
        training_pipeline.source_identity()


def test_pinned_config_and_model_identity(tmp_path):
    config = synthetic_config("xgboost")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert scheduled.read_config(path.as_uri(), sha, config["model_name"]) == config
    with pytest.raises(ValueError, match="identity"):
        scheduled.read_config(path.as_uri(), sha, "another_model")


def test_daily_passes_frozen_cutoff_and_never_forces_retry(tmp_path, monkeypatch, capsys):
    from recommend.train.comm import daily_publication

    config = synthetic_config("xgboost")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    bindings = {
        "ARTIFACT_ROOT": tmp_path.as_uri(),
        "MODEL_NAME": config["model_name"],
        "S3_KMS_KEY_ARN": "test",
        "CONFIG_URI": path.as_uri(),
        "CONFIG_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "DAILY_ROOT": "s3://test/input",
        "FROZEN_ROOT": "s3://test/frozen",
    }
    for key, value in bindings.items():
        monkeypatch.setenv("DOKU_TRAIN_" + key, value)
    received = {}

    def fake_daily(config, **kwargs):
        received.update(kwargs)
        return {"status": "rejected"}

    monkeypatch.setattr(daily_publication, "daily_run", fake_daily)
    monkeypatch.setattr(sys, "argv", ["scheduled", "daily", "--cutoff", "2026-10-12"])
    scheduled.main()
    assert received["cutoff"] == "2026-10-12" and "force" not in received
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "rejected"
    monkeypatch.setenv("DOKU_TRAIN_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="digest"):
        scheduled.main()
    assert json.loads(capsys.readouterr().out)["completed"] is False


@pytest.mark.parametrize("exists", [False, True])
def test_scoped_list_distinguishes_missing_from_denied(exists):
    from botocore.exceptions import ClientError

    store = ObjectStore("s3://test-bucket/models")
    store._client = boto3.client(
        "s3", region_name="us-east-2", aws_access_key_id="test", aws_secret_access_key="test"
    )
    with Stubber(store.client) as stub:
        stub.add_client_error(
            "get_object",
            service_error_code="AccessDenied",
            http_status_code=403,
            expected_params={"Bucket": "test-bucket", "Key": "models/registry.json"},
        )
        stub.add_response(
            "list_objects_v2",
            {"Contents": [{"Key": "models/registry.json"}] if exists else []},
            {"Bucket": "test-bucket", "Prefix": "models/registry.json", "MaxKeys": 1},
        )
        with pytest.raises(ClientError if exists else FileNotFoundError):
            store.get("registry.json")
        stub.assert_no_pending_responses()


def test_deployment_contract_matches_dkd_psm_and_fails_without_command():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    service = yaml.safe_load((root / "service.yml").read_text())
    assert re.fullmatch(r"([a-z][a-z0-9]*-){2}[a-z][a-z0-9]*", service["psm"])
    task = yaml.safe_load((root / "deploy/prod/task-definition.yml").read_text())
    container = task["containerDefinitions"][0]
    assert container["name"] == "{{PSM}}"
    assert container["logConfiguration"]["options"]["awslogs-group"].endswith(service["psm"])
    assert not (root / "deploy/dte").exists()
    dockerfile = (root / "build/Dockerfile").read_text()
    assert 'CMD ["python", "-m", "recommend.train.scheduled"]' in dockerfile


def test_cleanup_before_first_training_does_not_create_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOKU_TRAIN_ARTIFACT_ROOT", tmp_path.as_uri())
    monkeypatch.setenv("DOKU_TRAIN_MODEL_NAME", "test")
    monkeypatch.setenv("DOKU_TRAIN_S3_KMS_KEY_ARN", "test")
    monkeypatch.setattr(sys, "argv", ["scheduled", "cleanup"])
    scheduled.main()
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["dry_run"] and result["result"]["plan"] == []
    assert not (tmp_path / "test/publication.json").exists()
