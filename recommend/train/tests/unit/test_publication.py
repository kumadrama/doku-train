from __future__ import annotations

import hashlib
import itertools
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from recommend.train.comm.artifact_store import ConflictError, ObjectStore
from recommend.train.comm.daily_publication import choose_dates
from recommend.train.comm.publication import Registry
from recommend.train.comm.task_metrics import regression_auc
from recommend.train.export_model import gate


def test_concordance_matches_bruteforce():
    rng = np.random.default_rng(7)
    for n in range(1, 40):
        y, p = rng.integers(0, 5, size=(2, n))
        values = []
        for i, j in itertools.combinations(range(n), 2):
            if y[i] != y[j]:
                values.append(0.5 if p[i] == p[j] else float((p[i] - p[j]) * (y[i] - y[j]) > 0))
        assert (
            regression_auc(y, p) == pytest.approx(np.mean(values))
            if values
            else regression_auc(y, p) is None
        )


def test_window_has_gaps_and_natural_day_bound():
    assert choose_dates(["2026-10-01", "2026-10-03"], window_days=30) is None
    result = choose_dates(["2026-08-01", "2026-09-05", "2026-10-01", "2026-10-03"], window_days=30)
    assert result["train_dates"] == ["2026-09-05"]
    assert result["validation_dates"] == ["2026-10-01"]
    assert result["test_dates"] == ["2026-10-03"]


def test_gate_null_boundary_and_regression():
    config = {"tasks": {"x": {}}, "publication": {"x": {"metric": "rank", "minimum": 0.6}}}
    for value, passed in [(None, False), (0.59, False), (0.6, True), (0.9, True)]:
        assert gate(config, {"x": {"metrics": {"rank": {"value": value}}}})["passed"] == passed


def seed(registry, version, created, *, day="2026-10-01"):
    owner = "run-" + version
    registry.start(owner, version)
    key = f"test/{day}/{version}/manifest.json"
    registry.store.put(key, b"payload")
    registry.register(
        version,
        {
            "date": day,
            "created_at": created.isoformat(),
            "manifest_sha256": hashlib.sha256(b"payload").hexdigest(),
            "objects": {key: hashlib.sha256(b"payload").hexdigest()},
        },
        owner,
    )
    registry.publish(version, owner)
    registry.finish(owner, "published")
    return key


def test_pins_drain_rollback_and_deletion(tmp_path):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    now = datetime.now(UTC)
    old = seed(registry, "v1", now - timedelta(days=40))
    registry.pin("v1", "consumer-instance")
    seed(registry, "v2", now - timedelta(days=39), day="2026-10-02")
    later = now + timedelta(days=2)
    assert registry.cleanup(apply=True, now=later)["dry_run"]
    registry.consumer_ready()
    registry.cleanup(apply=True, now=later)
    assert registry.read()[0]["versions"]["v1"]["state"] == "draining"
    with pytest.raises(ValueError, match="not available"):
        registry.pin("v1", "new-instance")
    registry.cleanup(apply=True, now=later + timedelta(days=2))
    assert registry.store.get(old)[0] == b"payload"  # persistent pin, no timeout expiry
    registry.publish("v1", "operator", rollback=True)
    assert registry.read()[0]["current"] == "v1"
    registry.publish("v2", "operator", rollback=True)
    registry.cleanup(apply=True, now=later + timedelta(days=3))
    registry.pin("v1", "consumer-instance", release=True)
    registry.cleanup(apply=True, now=later + timedelta(days=5))
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"
    with pytest.raises(FileNotFoundError):
        registry.store.get(old)
    with pytest.raises(ValueError):
        registry.publish("v1", "operator", rollback=True)
    assert registry.read()[0]["versions"]["v2"]["state"] == "available"


def test_registry_duplicate_and_stale_cas(tmp_path):
    store = ObjectStore(tmp_path.as_uri())
    store.put("x", b"one")
    token = store.get("x")[1]
    store.put("x", b"two", expected=token)
    with pytest.raises(ConflictError):
        store.put("x", b"three", expected=token)
    with pytest.raises(ValueError):
        store.put("../escape", b"bad")
    registry = Registry(store, "test")
    assert registry.start("a", "same")
    with pytest.raises(ConflictError):
        registry.start("b", "different")
    registry.finish("a", "published")
    assert not registry.start("c", "same")
    assert registry.start("c", "same", force=True)


def test_concurrent_cas_recomputes_after_pin(tmp_path, monkeypatch):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    now = datetime.now(UTC)
    seed(registry, "v1", now - timedelta(days=40))
    seed(registry, "v2", now - timedelta(days=40), day="2026-10-02")
    registry.consumer_ready()
    original = registry.store.put
    once = True

    def racing(key, data, *, expected=None):
        nonlocal once
        if once:
            once = False
            registry.pin("v1", "racing-reader")
        return original(key, data, expected=expected)

    monkeypatch.setattr(registry.store, "put", racing)
    registry.cleanup(apply=True, now=now + timedelta(days=2))
    registry.cleanup(apply=True, now=now + timedelta(days=4))
    assert registry.read()[0]["versions"]["v1"]["pins"] == {
        "racing-reader": pytest.approx(
            registry.read()[0]["versions"]["v1"]["pins"]["racing-reader"]
        )
    }
    assert registry.read()[0]["versions"]["v1"]["state"] == "draining"


def test_s3_conditional_write_and_missing_are_not_permissions():
    import boto3
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test-only",
        aws_secret_access_key="test-only",
    )
    store = ObjectStore("s3://fixture-bucket/models")
    store._client = client
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {},
            {"Bucket": "fixture-bucket", "Key": "models/a", "Body": b"a", "IfNoneMatch": "*"},
        )
        store.put("a", b"a")
        stub.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params={
                "Bucket": "fixture-bucket",
                "Key": "models/a",
                "Body": b"b",
                "IfMatch": '"old"',
            },
        )
        with pytest.raises(ConflictError):
            store.put("a", b"b", expected='"old"')
        stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        with pytest.raises(FileNotFoundError):
            store.get("missing")
        stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)
        with pytest.raises(client.exceptions.ClientError):
            store.get("denied")
        stub.assert_no_pending_responses()


def test_delete_failure_keeps_fence_and_retries(tmp_path, monkeypatch):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    now = datetime.now(UTC)
    seed(registry, "v1", now - timedelta(days=40))
    seed(registry, "v2", now - timedelta(days=40), day="2026-10-02")
    registry.consumer_ready()
    registry.cleanup(apply=True, now=now + timedelta(days=2))
    original = registry.store.delete

    def failed(key):
        raise OSError("injected delete interruption")

    monkeypatch.setattr(registry.store, "delete", failed)
    with pytest.raises(OSError):
        registry.cleanup(apply=True, now=now + timedelta(days=4))
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleting"
    with pytest.raises(ValueError):
        registry.pin("v1", "reader")
    with pytest.raises(ValueError):
        registry.publish("v1", "operator", rollback=True)
    monkeypatch.setattr(registry.store, "delete", original)
    registry.cleanup(apply=True, now=now + timedelta(days=4))
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"


@pytest.mark.parametrize("error", [TimeoutError, ConflictError])
def test_lost_successful_put_response_is_reconciled(tmp_path, monkeypatch, error):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    original = registry.store.put

    def lost(key, data, *, expected=None):
        original(key, data, expected=expected)
        raise error("response lost after commit")

    monkeypatch.setattr(registry.store, "put", lost)
    assert registry.start("owner", "fingerprint")
    assert registry.read()[0]["active_run"] == "owner"
    registry.store.immutable("test/blob", b"exact content")
    assert registry.store.get("test/blob")[0] == b"exact content"
    registry.finish("owner", "failed")
    assert registry.read()[0]["active_run"] is None


def test_same_day_old_package_and_post_rollback_stale_publish_rejected(tmp_path):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    now = datetime.now(UTC)
    seed(registry, "first", now - timedelta(days=2))
    seed(registry, "newest", now)
    registry.publish("first", "operator", rollback=True)
    with pytest.raises(ValueError, match="stale"):
        seed(registry, "older", now - timedelta(days=1))
    assert registry.read()[0]["current"] == "first"


def test_shared_training_run_survives_local_cleanup(tmp_path):
    from recommend.train.comm.publication import cleanup_local

    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    now = datetime.now(UTC)
    seed(registry, "old", now - timedelta(days=40))
    seed(registry, "current", now - timedelta(days=39), day="2026-10-02")

    def same_run(state):
        for entry in state["versions"].values():
            entry["run_id"] = "shared"
        state["versions"]["old"]["state"] = "deleted"

    registry.update(same_run)
    local = tmp_path / "runs/shared"
    local.mkdir(parents=True)
    (local / "checkpoint").write_bytes(b"keep")
    cleanup_local(registry, tmp_path / "runs", tmp_path / "exports", apply=True)
    assert local.exists()

    def retire_all(state):
        state["versions"]["current"]["state"] = "deleted"
        state["current"] = None

    registry.update(retire_all)
    cleanup_local(registry, tmp_path / "runs", tmp_path / "exports", apply=True)
    assert not local.exists()
    registry.start("new-owner", "new")
    with pytest.raises(ConflictError, match="retired locally"):
        registry.register("new", {"run_id": "shared"}, "new-owner")


def test_two_cleaners_finish_idempotently(tmp_path, monkeypatch):
    registry = Registry(ObjectStore(tmp_path.as_uri()), "test")
    now = datetime.now(UTC)
    seed(registry, "old", now - timedelta(days=40))
    seed(registry, "current", now - timedelta(days=39), day="2026-10-02")
    registry.consumer_ready()
    registry.cleanup(apply=True, now=now + timedelta(days=2))
    original = registry.store.delete
    once = True

    def other_cleaner(key):
        nonlocal once
        if once:
            once = False
            registry.cleanup(apply=True, now=now + timedelta(days=4))
        original(key)

    monkeypatch.setattr(registry.store, "delete", other_cleaner)
    registry.cleanup(apply=True, now=now + timedelta(days=4))
    assert registry.read()[0]["versions"]["old"]["state"] == "deleted"


def test_active_local_reader_prevents_run_retirement(tmp_path):
    from recommend.train.comm.publication import cleanup_local
    from recommend.train.comm.run_lock import run_lock

    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    seed(registry, "v1", datetime.now(UTC) - timedelta(days=40))

    def deleted(state):
        state["versions"]["v1"]["state"] = "deleted"
        state["current"] = None

    registry.update(deleted)
    local = tmp_path / "runs/run-v1"
    local.mkdir(parents=True)
    with run_lock(local):
        cleanup_local(registry, tmp_path / "runs", tmp_path / "exports", apply=True)
        assert local.exists()
        assert "run-v1" not in registry.read()[0].get("retired_local_runs", [])
    cleanup_local(registry, tmp_path / "runs", tmp_path / "exports", apply=True)
    assert not local.exists()
