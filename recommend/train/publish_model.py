"""Publication use cases: exported package -> S3 -> Redis, or exact-version recovery."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.daily_publication import publish_package
from recommend.train.comm.publication import Registry, verify_objects
from recommend.train.comm.redis_distribution import Distribution


class DistributionFailed(RuntimeError):
    """Safe structured failure: never render Redis credentials or raw SDK messages."""

    def __init__(self, version: str, operation_id: str | None) -> None:
        self.result = {
            "status": "distribution_failed",
            "s3_publication": "published",
            "redis_distribution": "unknown",
            "model_version": version,
            "operation_id": operation_id,
        }
        super().__init__(json.dumps(self.result))


def sync_version(distribution: Distribution, version: str) -> dict[str, Any]:
    try:
        result = distribution.sync(version)
    except Exception:
        operation = None
        try:
            state = distribution.registry.read()[0]
            operation = state.get("distribution", {}).get("operation")
        except Exception:
            pass  # Storage unavailable: preserve the known S3 publication outcome.
        raise DistributionFailed(version, operation["id"] if operation else None) from None
    return {
        "status": "published",
        "s3_publication": "published",
        "redis_distribution": "committed",
        "model_version": version,
        "distribution": result,
    }


def publish_prepared(
    package: Path, registry: Registry, owner: str, *, run_dir: Path | None = None
) -> str:
    """S3 stage for an already-owned run; no training/export dependency."""
    return publish_package(package, registry, owner, run_dir=run_dir)


def execute_publish(
    package: Path,
    artifact_root: str,
    *,
    run_dir: Path | None = None,
    force: bool = False,
    distribution: Distribution | None = None,
) -> dict[str, Any]:
    raw = (package / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    version = manifest["model_version"]
    registry = Registry(ObjectStore(artifact_root), manifest["model_name"])
    state = registry.read()[0]
    if distribution is not None and (
        distribution.registry.name != registry.name
        or distribution.registry.store.root != registry.store.root
    ):
        raise ValueError("distribution binding mismatch")
    if state["format"] == "doku-publication/v2" and distribution is None:
        raise ValueError("Redis-enabled model requires distribution binding")
    existing = state["versions"].get(version)
    if existing is not None and existing["published"]:
        if existing["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("version collision")
        if existing["state"] in {"deleting", "deleted"}:
            raise ValueError("published version has retired")
        verify_objects(registry.store, existing)
        if distribution is not None:
            return sync_version(distribution, version)
        return {"status": "skipped", "reason": "already_published", "model_version": version}
    owner = uuid.uuid4().hex
    if not registry.start(owner, hashlib.sha256(raw).hexdigest(), force=force):
        return {"status": "skipped", "reason": "duplicate_package"}
    status = "failed"
    try:
        version = publish_prepared(package, registry, owner, run_dir=run_dir)
        status = "published"
    finally:
        registry.finish(owner, status)
    if distribution is not None:
        return sync_version(distribution, version)
    return {
        "status": status,
        "s3_publication": status,
        "redis_distribution": "disabled",
        "model_version": version,
    }


def distribution_from_environment(artifact_root: str, model_name: str) -> Distribution | None:
    """Explicit runtime bindings; no Redis endpoint or environment defaults."""
    import os

    from recommend.train.comm.redis_artifact_store import RedisArtifactStore

    url = os.environ.get("DOKU_TRAIN_REDIS_URL")
    keys = [
        "ENVIRONMENT",
        "CLUSTER",
        "MAX_FILE_BYTES",
        "MAX_PACKAGE_BYTES",
        "MAX_MODEL_BYTES",
        "TIMEOUT_SECONDS",
    ]
    if not url:
        if any("DOKU_TRAIN_REDIS_" + key in os.environ for key in keys):
            raise ValueError("incomplete Redis bindings")
        return None
    values = {key: os.environ["DOKU_TRAIN_REDIS_" + key] for key in keys}
    if values["CLUSTER"] not in {"true", "false"}:
        raise ValueError("Redis CLUSTER must be true or false")
    file_budget = int(values["MAX_FILE_BYTES"])
    package_budget = int(values["MAX_PACKAGE_BYTES"])
    model_budget = int(values["MAX_MODEL_BYTES"])
    if not 0 < file_budget <= package_budget <= model_budget:
        raise ValueError("invalid Redis byte budgets")
    redis = RedisArtifactStore.connect(
        url,
        cluster=values["CLUSTER"] == "true",
        max_file_bytes=file_budget,
        timeout_seconds=float(values["TIMEOUT_SECONDS"]),
    )
    return Distribution(
        Registry(ObjectStore(artifact_root, max_bytes=file_budget), model_name),
        redis,
        environment=values["ENVIRONMENT"],
        max_package_bytes=package_budget,
        max_model_bytes=model_budget,
    )


if __name__ == "__main__":
    # Share exactly the same CLI and use cases; this module defaults to its publish command.
    import sys

    from recommend.train.run import app

    app(["publish", *sys.argv[1:]])
