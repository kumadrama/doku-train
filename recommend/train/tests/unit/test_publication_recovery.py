"""Storage fault injection; fake model bytes test delivery, not native inference."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.daily_publication import publish_package
from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.publication import Registry, encode


def package_fixture(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    config = {"output": {"run_id": "training-run"}}
    (run / "resolved-config.json").write_bytes(encode(config))
    (run / "checkpoint.xgb.zip").write_bytes(b"training-checkpoint")
    files = {
        "models/0.ubj": b"synthetic-model",
        "feature-contract.json": b"{}",
        "fitted-feature-state/state.json": b"{}",
        "lineage.json": b"{}",
        "metrics.json": encode({"gate": {"passed": True}}),
    }
    for name, body in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    manifest = {
        "format": "doku-tree-artifact/v1",
        "model_name": "test",
        "model_version": "v1",
        "date": "2026-10-01",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": "training-run",
        "config_sha256": digest(config),
        "parity": {"status": "passed"},
        "tasks": {"task": {"file": "models/0.ubj"}},
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    (package / "manifest.json").write_bytes(encode(manifest))
    return package, run


def test_interrupted_daily_upload_resumes_with_exact_inventory(tmp_path, monkeypatch):
    package, run = package_fixture(tmp_path)
    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    registry.start("first", "input")
    original = registry.store.immutable

    def fail_training(key, data):
        if "/training/" in key:
            raise OSError("injected upload failure")
        original(key, data)

    monkeypatch.setattr(registry.store, "immutable", fail_training)
    with pytest.raises(OSError):
        publish_package(package, registry, "first", run_dir=run)
    assert registry.read()[0]["current"] is None
    assert registry.read()[0]["versions"]["v1"]["state"] == "uploading"
    with pytest.raises(FileNotFoundError):
        registry.store.get("test/2026-10-01/v1/manifest.json")
    registry.finish("first", "failed")
    registry.start("retry", "input")
    monkeypatch.setattr(registry.store, "immutable", original)
    with pytest.raises(ValueError, match="--run-dir"):
        publish_package(package, registry, "retry")
    assert publish_package(package, registry, "retry", run_dir=run) == "v1"
    registry.finish("retry", "published")
    assert registry.read()[0]["current"] == "v1"


def test_reserved_manifest_file_is_rejected_before_registration(tmp_path):
    package, _ = package_fixture(tmp_path)
    manifest = json.loads((package / "manifest.json").read_bytes())
    manifest["files"]["manifest.json"] = "a" * 64
    (package / "manifest.json").write_bytes(encode(manifest))
    registry = Registry(ObjectStore((tmp_path / "store").as_uri()), "test")
    registry.start("owner", "input")
    with pytest.raises(ValueError, match="reserved"):
        publish_package(package, registry, "owner")
    assert not registry.read()[0]["versions"]
