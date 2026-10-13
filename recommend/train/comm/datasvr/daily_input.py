"""Resolve mutable daily pointers once, before a run reads any feature records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from recommend.train.comm.datasvr.published_dataset import digest


def resolve_days(
    options: dict[str, Any], days: list[date], read: Callable[[str], bytes]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    prefix = options["daily_prefix"].rstrip("/") + "/"
    if not days or len(set(days)) != len(days):
        raise ValueError("daily input dates must be nonempty and unique")
    pointers: dict[str, bytes] = {}
    manifests: list[dict[str, Any]] = []
    sources = []
    contracts: dict[str, bytes] = {}
    for day in sorted(days):
        day_prefix = prefix + f"date={day}/"
        key = day_prefix + "_SUCCESS"
        body = read(key)
        pointers[key] = body
        p = json.loads(body)
        if p.get("pointer_version") != "doku-train-day/v1" or p.get("date") != str(day):
            raise ValueError("invalid daily pointer")
        uri = p["manifest_uri"]
        run = uri.removeprefix(day_prefix + "run_id=").removesuffix("/manifest.json")
        if uri != day_prefix + "run_id=" + run + "/manifest.json" or not run.isalnum():
            raise ValueError("daily pointer must name an immutable run within its date")
        raw = read(uri)
        if hashlib.sha256(raw).hexdigest() != p["manifest_sha256"]:
            raise ValueError("daily manifest checksum mismatch")
        m = json.loads(raw)
        if m.get("manifest_version") != "doku-train-dataset-manifest/v1" or digest(
            {k: v for k, v in m.items() if k != "content_sha256"}
        ) != m.get("content_sha256"):
            raise ValueError("daily manifest digest mismatch")
        if (
            m.get("projection_sha256") != options["projection_sha256"]
            or p["projection_sha256"] != options["projection_sha256"]
        ):
            raise ValueError("incompatible projection; backfill requested dates first")
        for field in ["feature_schema_version", "label_definition_version"]:
            if m[field] != options[field]:
                raise ValueError("incompatible daily contract version")
        if manifests and any(
            m[k] != manifests[0][k] for k in ["dataset_id", "label_config_version"]
        ):
            raise ValueError("incompatible daily dataset or label config")
        parts = m["partitions"]
        if not parts or sum(p["rows"] for p in parts) != m["row_count"]:
            raise ValueError("invalid daily row counts")
        for part in parts:
            if (
                part["event_time_from"] != f"{day}T00:00:00Z"
                or part["event_time_to"] != f"{day + timedelta(days=1)}T00:00:00Z"
            ):
                raise ValueError("daily manifest date mismatch")
            if not part["uri"].startswith(uri.rsplit("/", 1)[0] + "/") or "/.." in part["uri"]:
                raise ValueError("shard outside immutable daily run")
            if part["rows"] < 0 or len(part["sha256"]) != 64:
                raise ValueError("invalid daily shard")
        for name in ["features", "labels"]:
            content = read(uri.rsplit("/", 1)[0] + f"/contracts/{name}.json")
            if name in contracts and json.loads(content) != json.loads(contracts[name]):
                raise ValueError("incompatible daily contracts")
            if name == "features" and digest(json.loads(content)) != options["projection_sha256"]:
                raise ValueError("projection contract digest mismatch")
            contracts[name] = content
        manifests.append(m)
        sources.append({"uri": uri, "sha256": p["manifest_sha256"], "date": str(day)})
    # No cross-day transaction exists in S3. Reject concurrent pointer switches
    # during resolution, then use only the pinned immutable URIs for this run.
    if any(read(uri) != body for uri, body in pointers.items()):
        raise ValueError("daily pointer changed while freezing input; retry")
    parts = [p for m in manifests for p in m["partitions"]]
    if len({p["uri"] for p in parts}) != len(parts):
        raise ValueError("duplicate daily shards")
    result = {
        k: manifests[0][k]
        for k in [
            "manifest_version",
            "dataset_id",
            "feature_schema_version",
            "label_definition_version",
            "label_config_version",
            "projection_sha256",
        ]
    }
    result.update(
        dataset_version="frozen-" + digest(sources),
        partitions=parts,
        as_of_ms=max(m["as_of_ms"] for m in manifests),
        row_count=sum(m["row_count"] for m in manifests),
        source_manifests=sources,
        contract_sha256={k: hashlib.sha256(v).hexdigest() for k, v in contracts.items()},
    )
    result["content_sha256"] = digest(result)
    return result, contracts
