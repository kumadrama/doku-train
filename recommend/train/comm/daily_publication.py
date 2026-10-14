"""Prepare immutable multi-day inputs and upload registered publication artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.datasvr.daily_input import resolve_days
from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.comm.publication import Registry, encode, verify_objects


def read_uri(uri: str, max_bytes: int) -> bytes:
    parent, key = uri.rsplit("/", 1)
    return ObjectStore(parent, max_bytes=max_bytes).get(key)[0]


def choose_dates(days: list[str], *, window_days: int) -> dict[str, Any] | None:
    if window_days < 1:
        raise ValueError("positive training window required")
    available = sorted({date.fromisoformat(day) for day in days})
    if len(available) < 3:
        return None
    test, validation = available[-1], available[-2]
    train = [
        d for d in available[:-2] if validation - timedelta(days=window_days) <= d < validation
    ]
    if not train:
        return None
    result: dict[str, Any] = {"timezone": "UTC"}
    for name, values in [("train", train), ("validation", [validation]), ("test", [test])]:
        result[f"{name}_from"] = str(values[0])
        result[f"{name}_to"] = str(values[-1])
        result[f"{name}_dates"] = [str(d) for d in values]
    return result


def prepare(
    config: dict[str, Any], daily_root: str, frozen_root: str, cutoff: str, *, window_days: int
) -> dict[str, Any] | None:
    budget = config["execution"]["memory_limit_bytes"]
    source = ObjectStore(daily_root, max_bytes=budget)
    cutoff_day = date.fromisoformat(cutoff)
    available: list[str] = []
    observed: dict[str, bytes] = {}
    manifests: dict[str, Any] = {}
    for key in sorted(source.list(), reverse=True):
        elements = key.split("/")
        if len(elements) != 2 or elements[-1] != "_SUCCESS" or not elements[0].startswith("date="):
            continue
        day = elements[0][5:]
        parsed_day = date.fromisoformat(day)
        if parsed_day > cutoff_day:
            continue
        if len(available) >= 2 and parsed_day < date.fromisoformat(available[1]) - timedelta(
            days=window_days
        ):
            break
        body = source.get(key)[0]
        pointer = json.loads(body)
        if pointer.get("date") != day or pointer.get("pointer_version") != "doku-train-day/v1":
            raise ValueError("invalid daily pointer")
        uri = pointer["manifest_uri"]
        if not uri.startswith(source.root + f"/date={day}/run_id="):
            raise ValueError("daily manifest outside source prefix")
        raw = read_uri(uri, budget)
        if hashlib.sha256(raw).hexdigest() != pointer["manifest_sha256"]:
            raise ValueError("daily manifest checksum mismatch")
        manifest = json.loads(raw)
        if (
            digest({k: v for k, v in manifest.items() if k != "content_sha256"})
            != manifest["content_sha256"]
        ):
            raise ValueError("daily logical digest mismatch")
        observed[day] = body
        manifests[day] = manifest
        if manifest["row_count"] > 0:
            available.append(day)
    split = choose_dates(available, window_days=window_days)
    if split is None:
        return None
    days = [
        date.fromisoformat(d)
        for part in ["train", "validation", "test"]
        for d in split[f"{part}_dates"]
    ]
    first = manifests[str(days[-1])]
    options = {
        "daily_prefix": daily_root,
        **{
            k: first[k]
            for k in ["projection_sha256", "feature_schema_version", "label_definition_version"]
        },
    }
    frozen, contracts = resolve_days(options, days, lambda uri: read_uri(uri, budget))
    for selected_day in days:
        if source.get(f"date={selected_day}/_SUCCESS")[0] != observed[str(selected_day)]:
            raise ValueError("daily pointer changed during discovery")
    frozen_store = ObjectStore(frozen_root, max_bytes=budget)
    prefix = frozen["content_sha256"]
    for name, body in contracts.items():
        frozen_store.immutable(f"{prefix}/contracts/{name}.json", body)
    for selected_day in days:
        frozen_store.immutable(
            f"{prefix}/evidence/{selected_day}/pointer.json", observed[str(selected_day)]
        )
        uri = json.loads(observed[str(selected_day)])["manifest_uri"]
        raw = read_uri(uri, budget)
        if (
            hashlib.sha256(raw).hexdigest()
            != json.loads(observed[str(selected_day)])["manifest_sha256"]
        ):
            raise ValueError("immutable source changed")
        frozen_store.immutable(f"{prefix}/evidence/{selected_day}/manifest.json", raw)
    frozen_store.immutable(f"{prefix}/manifest.json", encode(frozen))
    result = copy.deepcopy(config)
    result["split"] = split
    result["dataset"] = {"manifest_uri": frozen_store.root + f"/{prefix}/manifest.json"}
    return validate_config(result)


def publish_package(
    package: Path, registry: Registry, owner: str, *, run_dir: Path | None = None
) -> str:
    raw = (package / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["format"] != "doku-tree-artifact/v1" or manifest["model_name"] != registry.name:
        raise ValueError("artifact identity mismatch")
    version = str(manifest["model_version"])
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", version):
        raise ValueError("invalid version")
    if date.fromisoformat(manifest["date"]).isoformat() != manifest["date"]:
        raise ValueError("artifact date must be canonical")
    from recommend.train.comm.publication import instant

    instant(manifest["created_at"])
    required = {
        "feature-contract.json",
        "fitted-feature-state/state.json",
        "metrics.json",
        "lineage.json",
    }
    if not required <= set(manifest["files"]) or not manifest["tasks"]:
        raise ValueError("incomplete artifact contract")
    for task in manifest["tasks"].values():
        if task["file"] not in manifest["files"] or not task["file"].startswith("models/"):
            raise ValueError("missing task model")
    base = f"{registry.name}/{manifest['date']}/{version}"
    files: dict[str, bytes] = {"manifest.json": raw}
    retained_bytes = len(raw)
    for name, sha in manifest["files"].items():
        registry.store.key(name)
        if name == "manifest.json" or name.startswith("training/"):
            raise ValueError("reserved artifact path")
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()) or path == package.resolve():
            raise ValueError("invalid artifact path")
        retained_bytes += path.stat().st_size
        if retained_bytes > registry.store.max_bytes // 2:
            raise ValueError("package exceeds upload memory budget")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != sha:
            raise ValueError("artifact checksum mismatch")
        files[name] = body
    report = json.loads(files["metrics.json"])
    if report["gate"]["passed"] is not True or manifest["parity"]["status"] != "passed":
        raise ValueError("artifact not approved")
    if run_dir is not None:
        from recommend.train.comm.run_lock import run_lock

        with run_lock(run_dir):
            archived_config = json.loads((run_dir / "resolved-config.json").read_text())
            if (
                archived_config["output"]["run_id"] != manifest["run_id"]
                or digest(archived_config) != manifest["config_sha256"]
            ):
                raise ValueError("training archive belongs to another run")
            for path in run_dir.rglob("*"):
                relative = str(path.relative_to(run_dir))
                if path.is_file() and (
                    relative.startswith("input/")
                    or relative
                    in {
                        "checkpoint.xgb.zip",
                        "resolved-config.json",
                        "status.json",
                        "metrics.json",
                        "test-metrics.json",
                        "test-evidence.json",
                        "lineage.json",
                        "fitted-feature-state/state.json",
                    }
                ):
                    retained_bytes += path.stat().st_size
                    if retained_bytes > registry.store.max_bytes // 2:
                        raise ValueError("training archive exceeds upload memory budget")
                    files["training/" + relative] = path.read_bytes()
    # A daily upload includes training evidence. Standalone retries must preserve
    # its exact registered inventory, including files missing after an interruption.
    existing = registry.read()[0]["versions"].get(version)
    if existing is not None and existing["manifest_sha256"] == hashlib.sha256(raw).hexdigest():
        for key, expected in existing["objects"].items():
            prefix = base + "/"
            if not key.startswith(prefix):
                raise ValueError("invalid registered inventory")
            relative = key[len(prefix) :]
            if relative not in files:
                if not relative.startswith("training/"):
                    raise ValueError("artifact inventory mismatch")
                try:
                    body = registry.store.get(key)[0]
                except FileNotFoundError as exc:
                    raise ValueError(
                        "retry requires --run-dir for missing training archive"
                    ) from exc
                if hashlib.sha256(body).hexdigest() != expected:
                    raise ValueError("training archive checksum mismatch")
                retained_bytes += len(body)
                if retained_bytes > registry.store.max_bytes // 2:
                    raise ValueError("retry archive exceeds upload memory budget")
                files[relative] = body
    entry = {
        "date": manifest["date"],
        "run_id": manifest["run_id"],
        "created_at": manifest["created_at"],
        "manifest_uri": registry.store.root + "/" + base + "/manifest.json",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "objects": {
            base + "/" + name: hashlib.sha256(body).hexdigest() for name, body in files.items()
        },
    }
    registry.register(version, entry, owner)
    for name in sorted(files, key=lambda name: (name == "manifest.json", name)):
        registry.store.immutable(base + "/" + name, files[name])
    verify_objects(registry.store, entry)
    registry.publish(version, owner)
    return version


def archive_unpublished_run(
    root: Path, config: dict[str, Any], registry: Registry, owner: str
) -> None:
    """Register immutable diagnostic artifacts before upload; never make them resolvable."""
    from datetime import UTC, datetime

    if owner in registry.read()[0]["versions"]:
        return  # an interrupted package upload already owns its exact object inventory
    files = {}
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if (
            path.is_file()
            and path.suffix in {".json", ".zip"}
            and not any(
                part.startswith("encoded-") or part.startswith("evaluation-")
                for part in path.relative_to(root).parts
            )
        ):
            if path.stat().st_size > registry.store.max_bytes:
                raise ValueError("diagnostic object exceeds storage budget")
            files["training/" + relative] = path.read_bytes()
    base = f"{registry.name}/{config['split']['test_to']}/{owner}"
    descriptor = encode(
        {
            "format": "doku-unpublished-run/v1",
            "run_id": owner,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        }
    )
    files["diagnostic.json"] = descriptor
    entry = {
        "date": config["split"]["test_to"],
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(descriptor).hexdigest(),
        "kind": "diagnostic",
        "objects": {base + "/" + k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    registry.register(owner, entry, owner)
    for key, body in files.items():
        registry.store.immutable(base + "/" + key, body)
