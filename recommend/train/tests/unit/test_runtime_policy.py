import io
import json
from unittest.mock import Mock

import pytest

from recommend.train import runtime_policy as policy


def document():
    return {
        **{k: {"enabled": True} for k in ("training", "publication", "cleanup")},
        "redis": {
            "enabled": True,
            "timeout_seconds": 10,
            "max_file_bytes": 10,
            "max_package_bytes": 20,
            "max_model_bytes": 30,
        },
    }


def test_fresh_session_and_closed_body(monkeypatch):
    monkeypatch.setenv("DOKU_ENV", "dte")
    client = Mock()
    client.start_configuration_session.return_value = {"InitialConfigurationToken": "test"}
    body = io.BytesIO(json.dumps(document()).encode())
    client.get_latest_configuration.return_value = {"Configuration": body}
    assert policy.read_policy(client=client) == document()
    assert body.closed
    assert client.start_configuration_session.call_args.kwargs["EnvironmentIdentifier"] == "dte"


@pytest.mark.parametrize("raw", [b"", b"{}", b"null", b"[]", b"not-json"])
def test_invalid_fails_closed(monkeypatch, raw):
    monkeypatch.setenv("DOKU_ENV", "prod")
    client = Mock()
    client.start_configuration_session.return_value = {"InitialConfigurationToken": "test"}
    client.get_latest_configuration.return_value = {"Configuration": io.BytesIO(raw)}
    with pytest.raises(ValueError, match="APPCONFIG"):
        policy.read_policy(client=client)


def test_redis_pause_blocks_publication_not_cleanup(monkeypatch):
    value = document()
    value["redis"]["enabled"] = False
    monkeypatch.setattr(policy, "read_policy", lambda: value)
    with pytest.raises(policy.OperationPaused, match="redis"):
        policy.check("s3://bucket/model", "publication")
    policy.check("s3://bucket/model", "cleanup")


def test_local_never_reads_cloud(monkeypatch):
    monkeypatch.setattr(policy, "read_policy", Mock(side_effect=AssertionError))
    policy.check("file:///tmp/model", "publication")


def test_daily_disabled_before_any_input_io(monkeypatch, tmp_path):
    from recommend.train import run

    value = document()
    value["training"]["enabled"] = False
    monkeypatch.setattr(policy, "read_policy", lambda: value)
    monkeypatch.setattr(run, "prepare", Mock(side_effect=AssertionError("input accessed")))
    result = run.daily_run(
        {},
        daily_root="s3://input",
        frozen_root="s3://frozen",
        cutoff="2026-10-15",
        artifact_root="s3://models",
        export_root=tmp_path,
    )
    assert result == {"status": "skipped", "reason": "training_disabled"}


def test_publish_disabled_before_package_or_registry_io(monkeypatch, tmp_path):
    from recommend.train.publish_model import execute_publish

    value = document()
    value["publication"]["enabled"] = False
    monkeypatch.setattr(policy, "read_policy", lambda: value)
    assert execute_publish(tmp_path / "missing", "s3://models") == {
        "status": "skipped",
        "reason": "publication_disabled",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", "false"),
        ("timeout_seconds", True),
        ("max_file_bytes", 100),
        ("max_model_bytes", 0),
    ],
)
def test_invalid_redis_policy(monkeypatch, field, value):
    monkeypatch.setenv("DOKU_ENV", "prod")
    data = document()
    data["redis"][field] = value
    client = Mock()
    client.start_configuration_session.return_value = {"InitialConfigurationToken": "test"}
    client.get_latest_configuration.return_value = {
        "Configuration": io.BytesIO(json.dumps(data).encode())
    }
    with pytest.raises(ValueError, match="APPCONFIG"):
        policy.read_policy(client=client)


def test_scheduled_cleanup_pause_no_store_and_no_success_metric(monkeypatch, capsys):
    import sys

    from recommend.train import scheduled

    data = document()
    data["cleanup"]["enabled"] = False
    monkeypatch.setattr(policy, "read_policy", lambda: data)
    monkeypatch.setenv("DOKU_TRAIN_ARTIFACT_ROOT", "s3://models/tree")
    monkeypatch.setattr(sys, "argv", ["scheduled", "cleanup"])
    scheduled.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "skipped"
    assert "completed" not in result


def test_admitted_publication_finishes_redis_without_second_gate(monkeypatch):
    from recommend.train.publish_model import sync_version

    monkeypatch.setattr(policy, "read_policy", Mock(side_effect=AssertionError("mid-commit read")))
    distribution = Mock()
    distribution.registry.store.root = "s3://models/tree"
    distribution.sync.return_value = {"status": "committed"}
    assert sync_version(distribution, "version", policy_checked=True)["status"] == "published"
    distribution.sync.assert_called_once_with("version")


def test_standalone_sync_pause_has_no_redis_calls(monkeypatch):
    from recommend.train.publish_model import sync_version

    data = document()
    data["redis"]["enabled"] = False
    monkeypatch.setattr(policy, "read_policy", lambda: data)
    distribution = Mock()
    distribution.registry.store.root = "s3://models/tree"
    assert sync_version(distribution, "version")["reason"] == "redis_disabled"
    distribution.sync.assert_not_called()


def test_cloud_binding_cannot_silently_fall_back_to_s3(monkeypatch):
    from recommend.train.publish_model import distribution_from_environment

    monkeypatch.setattr(policy, "read_policy", document)
    monkeypatch.delenv("DOKU_TRAIN_REDIS_URL", raising=False)
    monkeypatch.delenv("DOKU_TRAIN_REDIS_SECRET_ID", raising=False)
    with pytest.raises(ValueError, match="require Redis bindings"):
        distribution_from_environment("s3://models/tree", "tree")


def test_cross_environment_redis_rejected_before_secret_read(monkeypatch):
    from recommend.train.publish_model import distribution_from_environment

    monkeypatch.setattr(policy, "read_policy", document)
    monkeypatch.setenv("DOKU_ENV", "dte")
    monkeypatch.setenv("DOKU_TRAIN_REDIS_SECRET_ID", "prod-secret")
    monkeypatch.delenv("DOKU_TRAIN_REDIS_URL", raising=False)
    monkeypatch.setenv("DOKU_TRAIN_REDIS_ENVIRONMENT", "prod")
    monkeypatch.setenv("DOKU_TRAIN_REDIS_CLUSTER", "false")
    with pytest.raises(ValueError, match="does not match"):
        distribution_from_environment("s3://models/tree", "tree")
