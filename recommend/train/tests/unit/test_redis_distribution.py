"""Fault-injected delivery tests; model bytes are synthetic, never production data."""

import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from recommend.train.comm.artifact_store import ConflictError, ObjectStore
from recommend.train.comm.daily_publication import publish_package
from recommend.train.comm.publication import Registry, encode
from recommend.train.comm.redis_distribution import Distribution
from recommend.train.publish_model import DistributionFailed, execute_publish
from recommend.train.tests.unit.test_publication_recovery import package_fixture


class MemoryRedis:
    def __init__(self):
        self.data = {}
        self.fail_key = None
        self.fail_delete = None
        self.after_commit = None

    def get(self, key):
        return self.data.get(key)

    def immutable(self, key, data):
        if self.fail_key and self.fail_key in key:
            raise OSError("injected upload")
        if key in self.data and self.data[key] != data:
            raise ConflictError("immutable collision")
        self.data[key] = data

    def delete(self, key):
        if self.fail_delete and self.fail_delete in key:
            raise OSError("injected delete")
        self.data.pop(key, None)

    def commit(self, key, expected, value, ready):
        if self.data.get(key) not in [expected, value]:
            raise ConflictError("catalog changed")
        if any(self.get(k) != v for k, v in ready.items()):
            raise ConflictError("ready changed")
        self.data[key] = value
        if self.after_commit:
            self.after_commit()


def seed(registry, directory, version, day):
    directory.mkdir()
    package, run = package_fixture(directory)
    path = package / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.update(
        model_type="rerank_tree",
        model_version=version,
        date=day,
        created_at=(datetime.now(UTC) - timedelta(days=50)).isoformat(),
    )
    path.write_bytes(encode(manifest))
    registry.start(version, "input-" + version)
    publish_package(package, registry, version, run_dir=run)
    registry.finish(version, "published")
    return package


@pytest.fixture
def setup(tmp_path):
    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    redis = MemoryRedis()
    distribution = Distribution(
        registry, redis, environment="test", max_package_bytes=100000, max_model_bytes=1000000
    )
    package = seed(registry, tmp_path / "first", "v1", "2026-10-01")
    return registry, redis, distribution, package


def test_serving_subset_and_version_contract(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    state = registry.read()[0]
    assert state["format"] == "doku-publication/v2"
    before = json.loads(redis.get(distribution.catalog_key))
    assert before["current"]["version"] == "v1"
    assert before["dates"]["2026-10-01"] == before["current"]
    descriptor = state["distribution"]["versions"]["v1"]["descriptor"]
    assert set(descriptor["required_files"]) == {
        "manifest.json",
        "models/0.ubj",
        "feature-contract.json",
        "fitted-feature-state/state.json",
    }
    assert not any("training" in k or "metrics" in k for k in redis.data)
    # Feature service pins and encodes v1. A concurrent update cannot alter that RPC context.
    registry.pin("v1", "features-instance")
    request = dict(before["current"])
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    assert json.loads(redis.get(distribution.catalog_key))["current"]["version"] == "v2"
    registry.pin(request["version"], "predict-instance")
    distribution.verify(request["version"])
    assert request["manifest_sha256"] == state["versions"]["v1"]["manifest_sha256"]
    with pytest.raises(ConflictError, match="coordinated cleanup"):
        registry.cleanup(apply=True)
    with pytest.raises(ConflictError, match="coordinated publication"):
        registry.publish("v1", "operator", rollback=True)


def test_incomplete_upload_preserves_old_catalog_and_requires_owned_resume(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    old = redis.get(distribution.catalog_key)
    package = seed(registry, tmp_path / "second", "v2", "2026-10-02")
    redis.fail_key = "v:v2:file:models/"
    with pytest.raises(DistributionFailed) as exc:
        execute_publish(package, registry.store.root, distribution=distribution)
    assert exc.value.result["s3_publication"] == "published"
    assert registry.read()[0]["current"] == "v2"
    assert redis.get(distribution.catalog_key) == old
    op = registry.read()[0]["distribution"]["operation"]
    assert op["id"] == exc.value.result["operation_id"]
    with pytest.raises(ConflictError):
        registry.start("next", "next")
    with pytest.raises(ValueError, match="confirm"):
        distribution.resume(op["id"], confirmed_stopped=False)
    with pytest.raises(ConflictError):
        distribution.resume("wrong-operation", confirmed_stopped=True)
    redis.fail_key = None
    distribution.resume(op["id"], confirmed_stopped=True)
    assert json.loads(redis.get(distribution.catalog_key))["current"]["version"] == "v2"
    assert registry.read()[0]["distribution"]["operation"] is None


def test_redis_commit_succeeded_but_s3_confirmation_failed(setup, monkeypatch):
    registry, redis, distribution, _ = setup
    original = registry.update
    fail = False

    def update(operation):
        if fail:
            raise OSError("S3 unavailable after Redis commit")
        return original(operation)

    def after_commit():
        nonlocal fail
        fail = True

    redis.after_commit = after_commit
    monkeypatch.setattr(registry, "update", update)
    with pytest.raises(OSError):
        distribution.sync("v1")
    op = registry.read()[0]["distribution"]["operation"]
    assert json.loads(redis.get(distribution.catalog_key))["operation_id"] == op["id"]
    fail = False
    redis.after_commit = None
    distribution.resume(op["id"], confirmed_stopped=True)
    assert registry.read()[0]["distribution"]["operation"] is None


def test_rollback_is_new_generation_and_stale_sync_cannot_undo_it(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    before = json.loads(redis.get(distribution.catalog_key))
    distribution.sync("v1", rollback=True)
    after = json.loads(redis.get(distribution.catalog_key))
    assert after["generation"] > before["generation"]
    assert after["current"]["version"] == registry.read()[0]["current"] == "v1"
    with pytest.raises(ValueError, match="exact S3 current"):
        distribution.sync("v2")
    with pytest.raises(ConflictError):
        redis.commit(distribution.catalog_key, encode(before), encode(before), {})
    assert registry.read()[0]["publication_high_watermark"][0] == "2026-10-02"


def test_redis_loss_and_corruption_rebuild_requires_explicit_fencing(setup):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    old_epoch = registry.read()[0]["distribution"]["epoch"]
    redis.data.clear()
    with pytest.raises(ConflictError, match="rebuild"):
        distribution.sync("v1")
    with pytest.raises(ConflictError):
        distribution.cleanup(apply=True)
    with pytest.raises(ValueError, match="confirm"):
        distribution.rebuild(confirmed_stopped=False)
    distribution.rebuild(confirmed_stopped=True)
    assert registry.read()[0]["distribution"]["epoch"] != old_epoch
    key = distribution.prefix + "v:v1:file:models/0.ubj"
    redis.data[key] = b"corrupt"
    with pytest.raises(ValueError, match="corrupt"):
        distribution.sync("v1")
    distribution.rebuild(confirmed_stopped=True)
    distribution.verify("v1")


def test_cleanup_protects_both_currents_then_drains_pins_and_deletes_redis_first(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    now = datetime.now(UTC) + timedelta(days=4)
    assert all(s["action"] == "current" for s in distribution.cleanup(now=now)["plan"])
    distribution.sync("v2")
    registry.pin("v1", "feature-instance")
    registry.pin("v1", "predict-instance")
    before = registry.store.get(registry.key)
    assert distribution.cleanup(apply=True, now=now)["dry_run"]
    assert registry.store.get(registry.key) == before
    registry.consumer_ready()
    assert distribution.cleanup(apply=True, now=now)["dry_run"]
    distribution.protocol_ready()
    distribution.cleanup(apply=True, now=now)
    state = registry.read()[0]
    assert state["distribution"]["operation"] is None  # Do not block daily publishing for 24h.
    assert state["versions"]["v1"]["state"] == "draining"
    assert "2026-10-01" not in json.loads(redis.get(distribution.catalog_key))["dates"]
    with pytest.raises(ValueError):
        registry.pin("v1", "new-instance")
    assert distribution.cleanup(now=now)["plan"][0]["action"] == "active_pins"
    registry.pin("v1", "feature-instance", release=True)
    assert distribution.cleanup(now=now)["plan"][0]["action"] == "active_pins"
    registry.pin("v1", "predict-instance", release=True)
    redis.fail_delete = "v:v1:file:models"
    with pytest.raises(OSError):
        distribution.cleanup(apply=True, now=now)
    state = registry.read()[0]
    assert state["versions"]["v1"]["state"] == "deleting"
    for key in state["versions"]["v1"]["objects"]:
        assert registry.store.get(key)[0]  # S3 intact until all Redis files are gone.
    with pytest.raises(ConflictError):
        distribution.sync("v1", rollback=True)
    redis.fail_delete = None
    distribution.resume(state["distribution"]["operation"]["id"], confirmed_stopped=True)
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"
    assert not any("v:v1:" in k for k in redis.data)
    distribution.verify("v2")
    assert registry.read()[0]["current"] == "v2"


def test_s3_delete_interruption_and_rollback_during_drain(setup, tmp_path, monkeypatch):
    registry, _redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    registry.consumer_ready()
    distribution.protocol_ready()
    now = datetime.now(UTC) + timedelta(days=4)
    distribution.cleanup(apply=True, now=now)
    distribution.sync("v1", rollback=True)
    assert registry.read()[0]["versions"]["v1"]["state"] == "available"
    assert distribution.cleanup(now=now)["plan"][0]["action"] == "current"
    distribution.sync("v2", rollback=True)
    distribution.cleanup(apply=True, now=now)
    original = registry.store.delete
    failures = []

    def fail_once(key):
        original(key)
        if not failures:
            failures.append(key)
            raise OSError("S3 deletion response lost")

    monkeypatch.setattr(registry.store, "delete", fail_once)
    with pytest.raises(OSError):
        distribution.cleanup(apply=True, now=now)
    state = registry.read()[0]
    assert state["versions"]["v1"]["state"] == "deleting"
    distribution.resume(state["distribution"]["operation"]["id"], confirmed_stopped=True)
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"


def test_environment_and_budget_fail_closed(setup):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    other = Distribution(
        registry, redis, environment="another", max_package_bytes=10, max_model_bytes=20
    )
    with pytest.raises(ValueError, match="environment"):
        other.sync("v1")
    distribution.max_package_bytes = 10
    with pytest.raises(ValueError, match="budget"):
        distribution.sync("v1")


def test_source_corruption_never_exposes_catalog(setup):
    registry, redis, distribution, _ = setup
    path = registry.store.path("test/2026-10-01/v1/models/0.ubj")
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        distribution.sync("v1")
    assert redis.get(distribution.catalog_key) is None


@pytest.mark.parametrize("mode", ["published", "rejected", "interrupted"])
def test_duplicate_daily_still_distributes_without_train_or_export(setup, monkeypatch, mode):
    from recommend.train import run
    from recommend.train.comm.datasvr.published_dataset import digest

    registry, redis, distribution, _ = setup
    from recommend.train import train_rerank

    monkeypatch.setattr(train_rerank, "execute_train", lambda *a: pytest.fail("retrained"))
    config = {
        "model_name": "test",
        "model_type": "rerank_tree",
        "publication": {},
        "daily": {"training_window_days": 30},
        "dataset": {"manifest_uri": "file:///unused"},
        "execution": {"memory_limit_bytes": 1000000, "cache_dir": "unused"},
        "output": {"root": "/unused"},
    }
    monkeypatch.setattr(run, "validate_config", lambda c: c)
    monkeypatch.setattr(run, "prepare", lambda *args, **kwargs: copy.deepcopy(config))
    monkeypatch.setattr(run, "read_uri", lambda *args: b"manifest")
    monkeypatch.setattr(run, "source_identity", lambda: {"source": "test"})
    logical = {k: v for k, v in config.items() if k != "output"}
    logical["execution"] = {"memory_limit_bytes": 1000000}
    import hashlib

    fingerprint = digest(
        {
            "config": logical,
            "source": {"source": "test"},
            "manifest": hashlib.sha256(b"manifest").hexdigest(),
        }
    )
    registry.update(lambda s: s["runs"]["v1"].update(fingerprint=fingerprint))
    args = dict(
        daily_root="unused",
        frozen_root="unused",
        cutoff="2026-10-01",
        artifact_root=registry.store.root,
        export_root=None,
        distribution=distribution,
    )
    if mode == "rejected":
        registry.update(
            lambda s: (
                s["runs"]["v1"].update(status="rejected"),
                s["versions"]["v1"].update(published=False),
            )
        )
        monkeypatch.setattr(distribution, "sync", lambda *a, **kw: pytest.fail("rejected sync"))
        assert run.daily_run(config, **args)["reason"] == "duplicate_input"
        return
    if mode == "interrupted":
        redis.fail_key = "file:models"
        with pytest.raises(DistributionFailed) as failure:
            run.daily_run(config, **args)
        redis.fail_key = None
        # A duplicate trigger must report the pending operation, not retrain or silently skip.
        with pytest.raises(DistributionFailed):
            run.daily_run(config, **args)
        distribution.resume(failure.value.result["operation_id"], confirmed_stopped=True)
    result = run.daily_run(config, **args)
    assert result["redis_distribution"] == "committed"
    assert run.daily_run(config, **args)["distribution"]["unchanged"]


def test_stage_and_cancel_leave_online_catalog_unchanged(setup, tmp_path):
    registry, redis, distribution, _ = setup
    assert distribution.sync("v1", stage_only=True)["status"] == "staged"
    assert redis.get(distribution.catalog_key) is None
    distribution.verify("v1")
    distribution.sync("v1")
    before = redis.get(distribution.catalog_key)
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    redis.fail_key = "v:v2:file:models"
    with pytest.raises(OSError):
        distribution.sync("v2")
    op = registry.read()[0]["distribution"]["operation"]
    with pytest.raises(ValueError, match="confirm"):
        distribution.cancel(op["id"], confirmed_stopped=False)
    distribution.cancel(op["id"], confirmed_stopped=True)
    assert redis.get(distribution.catalog_key) == before
    assert registry.read()[0]["distribution"]["operation"] is None
    redis.fail_key = None
    distribution.sync("v2")


def test_replacement_grace_counts_redis_current_even_when_date_is_retained(setup, tmp_path):
    registry, _, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    registry.update(
        lambda s: s["versions"]["v1"].update(
            replaced_at=(datetime.now(UTC) - timedelta(days=5)).isoformat()
        )
    )
    distribution.sync("v2")
    assert distribution.cleanup()["plan"][0]["action"] == "replacement_grace"
    assert registry.read()[0]["distribution"]["versions"]["v1"]["replaced_at"]


def test_delete_commit_lost_confirmation_is_resumable(setup, tmp_path, monkeypatch):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    registry.consumer_ready()
    distribution.protocol_ready()
    now = datetime.now(UTC) + timedelta(days=4)
    distribution.cleanup(apply=True, now=now)
    assert distribution.cleanup()["plan"][0]["action"] in {"drain_grace", "replacement_grace"}
    original = registry.update
    fail = False

    def update(operation):
        if fail:
            raise OSError("lost confirmation")
        return original(operation)

    def committed():
        nonlocal fail
        fail = True

    monkeypatch.setattr(registry, "update", update)
    redis.after_commit = committed
    with pytest.raises(OSError):
        distribution.cleanup(apply=True, now=now)
    state = registry.read()[0]
    assert state["versions"]["v1"]["state"] == "deleted"
    redis.after_commit = None
    fail = False
    distribution.resume(state["distribution"]["operation"]["id"], confirmed_stopped=True)
    assert registry.read()[0]["distribution"]["operation"] is None


def test_pin_racing_delete_acquisition_invalidates_plan(setup, tmp_path, monkeypatch):
    registry, _, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    registry.consumer_ready()
    distribution.protocol_ready()
    original = distribution._begin

    def concurrent_pin(*args, **kwargs):
        registry.pin("v1", "late-consumer")
        return original(*args, **kwargs)

    monkeypatch.setattr(distribution, "_begin", concurrent_pin)
    with pytest.raises(ConflictError, match="changed while planning"):
        distribution.cleanup(apply=True, now=datetime.now(UTC) + timedelta(days=4))
    assert registry.read()[0]["versions"]["v1"]["state"] == "available"


def test_total_budget_failure_preserves_catalog_and_registered_files(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    old = redis.get(distribution.catalog_key)
    distribution.max_model_bytes = registry.read()[0]["distribution"]["versions"]["v1"]["bytes"]
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    with pytest.raises(ValueError, match="retention exceeds budget"):
        distribution.sync("v2")
    assert redis.get(distribution.catalog_key) == old
    assert not any("v:v2:" in key for key in redis.data)


@pytest.mark.parametrize("seeded", [False, True])
def test_cleanup_before_first_sync_is_read_only(tmp_path, seeded):
    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    redis = MemoryRedis()
    distribution = Distribution(
        registry, redis, environment="test", max_package_bytes=100000, max_model_bytes=1000000
    )
    if seeded:
        seed(registry, tmp_path / "package", "v1", "2026-10-01")
    before = registry.read()
    assert distribution.cleanup(apply=True)["reason"] == "distribution_not_initialized"
    assert registry.read() == before
    assert not redis.data


@pytest.mark.parametrize(
    "url,timeout",
    [
        ("redis://remote.example/0", 3),
        ("rediss://remote.example/0?ssl_cert_reqs=none", 3),
        ("redis://127.0.0.1/0", float("nan")),
        ("redis://127.0.0.1/0", 0),
    ],
)
def test_unsafe_transport_options_rejected_without_connecting(url, timeout):
    from recommend.train.comm.redis_artifact_store import RedisArtifactStore

    with pytest.raises(ValueError):
        RedisArtifactStore.connect(url, cluster=False, max_file_bytes=100, timeout_seconds=timeout)


def test_ready_deletions_are_not_starved_by_new_drain_candidates(setup, tmp_path):
    registry, _, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    registry.consumer_ready()
    distribution.protocol_ready()
    future = datetime.now(UTC) + timedelta(days=4)
    distribution.cleanup(apply=True, now=future)
    seed(registry, tmp_path / "third", "v3", "2026-10-03")
    distribution.sync("v3")
    result = distribution.cleanup(apply=True, now=future)
    assert result["executed"] == "delete"
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"


@pytest.mark.parametrize("failure", ["upload", "confirmation"])
@pytest.mark.parametrize("corrupt_catalog", [False, True])
def test_pending_publication_can_recover_after_redis_loss(
    setup, tmp_path, failure, corrupt_catalog
):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    if failure == "upload":
        redis.fail_key = "v:v2:file:models"
    else:
        redis.after_commit = lambda: (_ for _ in ()).throw(OSError("confirmation lost"))
    with pytest.raises(OSError):
        distribution.sync("v2")
    op = registry.read()[0]["distribution"]["operation"]
    epoch = op["target"]["epoch"]
    redis.fail_key = redis.after_commit = None
    redis.data.clear()
    if corrupt_catalog:
        redis.data[distribution.catalog_key] = b"\xff\x00invalid-json"
    distribution.resume(op["id"], confirmed_stopped=True, rebuild_redis=True)
    catalog = json.loads(redis.get(distribution.catalog_key))
    assert catalog["epoch"] != epoch
    assert catalog["generation"] > op["target"]["generation"]
    assert catalog["current"]["version"] == "v2"
    distribution.verify("v1")  # Restore the complete retained catalog, not just the new package.
    distribution.verify("v2")
    assert registry.read()[0]["distribution"]["operation"] is None


def test_catalog_change_during_redis_deletion_blocks_s3_and_recovers(setup, tmp_path, monkeypatch):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    stale = redis.get(distribution.catalog_key)
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    registry.consumer_ready()
    distribution.protocol_ready()
    future = datetime.now(UTC) + timedelta(days=4)
    distribution.cleanup(apply=True, now=future)
    original = redis.delete

    def failover(key):
        original(key)
        redis.data[distribution.catalog_key] = stale

    monkeypatch.setattr(redis, "delete", failover)
    with pytest.raises(ConflictError, match="catalog changed"):
        distribution.cleanup(apply=True, now=future)
    state = registry.read()[0]
    for key in state["versions"]["v1"]["objects"]:
        assert registry.store.get(key)[0]
    monkeypatch.setattr(redis, "delete", original)
    distribution.resume(
        state["distribution"]["operation"]["id"], confirmed_stopped=True, rebuild_redis=True
    )
    assert registry.read()[0]["versions"]["v1"]["state"] == "deleted"
    catalog = json.loads(redis.get(distribution.catalog_key))
    assert catalog["current"]["version"] == "v2"
    assert "2026-10-01" not in catalog["dates"]
    distribution.verify("v2")


@pytest.mark.parametrize("kind", ["stage", "rollback", "drain"])
def test_interrupted_control_operations_rebuild_without_changing_intent(setup, tmp_path, kind):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    if kind != "stage":
        distribution.sync("v2")
    if kind == "stage":
        redis.fail_key = "v:v2:file:models"
    else:

        def fail_commit():
            raise OSError("commit interrupted")

        redis.after_commit = fail_commit
        if kind == "drain":
            registry.consumer_ready()
            distribution.protocol_ready()
    with pytest.raises(OSError):
        if kind == "stage":
            distribution.sync("v2", stage_only=True)
        elif kind == "rollback":
            distribution.sync("v1", rollback=True)
        else:
            distribution.cleanup(apply=True, now=datetime.now(UTC) + timedelta(days=4))
    op = registry.read()[0]["distribution"]["operation"]
    redis.fail_key = redis.after_commit = None
    redis.data.clear()
    with pytest.raises(ValueError, match="confirm"):
        distribution.resume(op["id"], confirmed_stopped=False, rebuild_redis=True)
    distribution.resume(op["id"], confirmed_stopped=True, rebuild_redis=True)
    catalog = json.loads(redis.get(distribution.catalog_key))
    assert catalog["current"]["version"] == ("v2" if kind == "drain" else "v1")
    if kind == "drain":
        assert "2026-10-01" not in catalog["dates"]
        assert registry.read()[0]["versions"]["v1"]["state"] == "draining"
    else:
        distribution.verify("v1")
        distribution.verify("v2")


def test_deletion_inventory_cannot_escape_version_prefix(setup, tmp_path):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    seed(registry, tmp_path / "second", "v2", "2026-10-02")
    distribution.sync("v2")
    unrelated = distribution.prefix + "v:v2:file:models/0.ubj"
    original = redis.get(unrelated)
    registry.consumer_ready()
    distribution.protocol_ready()
    future = datetime.now(UTC) + timedelta(days=4)
    distribution.cleanup(apply=True, now=future)
    registry.update(
        lambda s: s["distribution"]["versions"]["v1"]["objects"].update({unrelated: "wrong-owner"})
    )
    with pytest.raises(ValueError, match="unowned Redis key"):
        distribution.cleanup(apply=True, now=future)
    assert redis.get(unrelated) == original
    for key in registry.read()[0]["versions"]["v1"]["objects"]:
        assert registry.store.get(key)[0]


def test_binary_corrupt_catalog_can_be_rebuilt_without_guessing_version(setup):
    registry, redis, distribution, _ = setup
    distribution.sync("v1")
    old = registry.read()[0]["distribution"]["epoch"]
    redis.data[distribution.catalog_key] = b"\xff\x00broken-json"
    with pytest.raises(ConflictError):
        distribution.cleanup(apply=True)
    distribution.rebuild(confirmed_stopped=True)
    catalog = json.loads(redis.get(distribution.catalog_key))
    assert catalog["epoch"] != old
    assert catalog["current"]["version"] == "v1"
    distribution.verify("v1")
