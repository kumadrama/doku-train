"""Real Redis Lua and binary transfer; CI supplies an isolated Redis service."""

import json
import os
import subprocess
import sys
import uuid

import pytest
from redis.cluster import key_slot

from recommend.train.comm.artifact_store import ConflictError, ObjectStore
from recommend.train.comm.publication import Registry
from recommend.train.comm.redis_artifact_store import RedisArtifactStore
from recommend.train.comm.redis_distribution import Distribution
from recommend.train.run import daily_run
from recommend.train.tests.smoke.test_tree_publication import daily_fixture


@pytest.fixture
def transport():
    url = os.environ.get("DOKU_TEST_REDIS_URL")
    if not url:
        pytest.skip("DOKU_TEST_REDIS_URL required; enabled in CI")
    store = RedisArtifactStore.connect(
        url, cluster=False, max_file_bytes=10000000, timeout_seconds=3
    )
    store.client.ping()
    yield store
    store.client.close()


def test_atomic_catalog_guards_and_ack_loss(transport, monkeypatch):
    prefix = "test:{" + uuid.uuid4().hex + "}:"
    catalog, ready, payload = [prefix + part for part in ["catalog", "ready", "payload"]]
    assert len({key_slot(k.encode()) for k in [catalog, ready, payload]}) == 1
    try:
        transport.immutable(payload, b"\x00\xffbinary")
        assert transport.get(payload) == b"\x00\xffbinary"
        assert transport.client.ttl(payload) == -1
        with pytest.raises(ConflictError):
            transport.immutable(payload, b"different")
        with pytest.raises(ConflictError):
            transport.commit(catalog, None, b"one", {ready: b"ok"})
        assert transport.get(catalog) is None
        transport.immutable(ready, b"ok")
        transport.commit(catalog, None, b"one", {ready: b"ok"})
        original = transport.client.eval

        def lose_ack(*args):
            original(*args)
            raise TimeoutError("lost successful reply")

        monkeypatch.setattr(transport.client, "eval", lose_ack)
        transport.commit(catalog, b"one", b"two", {ready: b"ok"})
        assert transport.get(catalog) == b"two"
        monkeypatch.setattr(transport.client, "eval", original)
        with pytest.raises(ConflictError):
            transport.commit(catalog, b"one", b"stale", {ready: b"ok"})
        assert transport.get(catalog) == b"two"
        transport.delete(ready)
        with pytest.raises(ConflictError):
            transport.commit(catalog, b"two", b"three", {ready: b"ok"})
        assert transport.get(catalog) == b"two"
    finally:
        for key in [catalog, ready, payload]:
            transport.delete(key)


def test_cpu_training_to_redis_and_duplicate_recovery(tmp_path, transport, monkeypatch):
    from recommend.train import train_rerank

    config, daily = daily_fixture(tmp_path)
    registry = Registry(ObjectStore((tmp_path / "artifacts").as_uri()), config["model_name"])
    distribution = Distribution(
        registry,
        transport,
        environment="test-" + uuid.uuid4().hex,
        max_package_bytes=20000000,
        max_model_bytes=40000000,
    )
    args = dict(
        daily_root=daily.root,
        frozen_root=(tmp_path / "frozen").as_uri(),
        cutoff="2026-10-08",
        artifact_root=registry.store.root,
        export_root=tmp_path / "exports",
        distribution=distribution,
    )
    try:
        result = daily_run(config, **args)
        assert result["redis_distribution"] == "committed"
        version = result["model_version"]
        distribution.verify(version)
        catalog = json.loads(transport.get(distribution.catalog_key))
        assert catalog["current"]["version"] == version
        monkeypatch.setattr(train_rerank, "execute_train", lambda *a: pytest.fail("retrained"))
        assert daily_run(config, **args)["distribution"]["unchanged"]
        assert distribution.cleanup(apply=True)["dry_run"]
        record = registry.read()[0]["distribution"]["versions"][version]
        # Load the exact bytes fetched from Redis with the real XGBoost parser.
        for index, (path, file) in enumerate(record["descriptor"]["required_files"].items()):
            if path.endswith(".ubj"):
                downloaded = tmp_path / f"downloaded-{index}.ubj"
                downloaded.write_bytes(transport.get(file["key"]))
                # Keep XGBoost's native OpenMP runtime out of the PyTorch test process on macOS.
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys,xgboost as x; b=x.Booster(); b.load_model(sys.argv[1]); "
                        "assert b.num_features()>0",
                        str(downloaded),
                    ],
                    check=True,
                    timeout=30,
                )
    finally:
        for record in registry.read()[0].get("distribution", {}).get("versions", {}).values():
            for key in record["objects"]:
                transport.delete(key)
        transport.delete(distribution.catalog_key)


def test_real_pending_commit_recovery_after_scoped_redis_loss(tmp_path, transport, monkeypatch):
    from recommend.train.tests.unit.test_redis_distribution import seed

    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    distribution = Distribution(
        registry,
        transport,
        environment="test-" + uuid.uuid4().hex,
        max_package_bytes=100000,
        max_model_bytes=1000000,
    )
    try:
        seed(registry, tmp_path / "first", "v1", "2026-10-01")
        distribution.sync("v1")
        seed(registry, tmp_path / "second", "v2", "2026-10-02")
        commit = transport.commit

        def interrupted(*args):
            commit(*args)
            raise OSError("process interrupted after Redis commit")

        monkeypatch.setattr(transport, "commit", interrupted)
        with pytest.raises(OSError):
            distribution.sync("v2")
        state = registry.read()[0]
        op = state["distribution"]["operation"]
        # Simulate cache loss only within this test's owned namespace; never FLUSHDB.
        for record in state["distribution"]["versions"].values():
            for key in record["objects"]:
                transport.delete(key)
        transport.delete(distribution.catalog_key)
        monkeypatch.setattr(transport, "commit", commit)
        distribution.resume(op["id"], confirmed_stopped=True, rebuild_redis=True)
        assert json.loads(transport.get(distribution.catalog_key))["epoch"] != op["target"]["epoch"]
        distribution.verify("v1")
        distribution.verify("v2")
    finally:
        for record in registry.read()[0].get("distribution", {}).get("versions", {}).values():
            for key in record["objects"]:
                transport.delete(key)
        transport.delete(distribution.catalog_key)
