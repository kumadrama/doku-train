"""Resolve mutable daily pointers once, before a run reads any feature records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from recommend.train.comm.datasvr.daily_contracts import (
    digest,
    label_versions,
    merge_contracts,
    merged_dataset_identity,
    validate_contracts,
)


def resolve_days(
    options: dict[str, Any], days: list[date], read: Callable[[str], bytes]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    prefix = options["daily_prefix"].rstrip("/") + "/"
    if not days or len(set(days)) != len(days):
        raise ValueError("daily input dates must be nonempty and unique")
    pointers: dict[str, bytes] = {}
    manifests: list[dict[str, Any]] = []
    sources = []
    bundles: list[dict[str, Any]] = []
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
        if not m.get("projection_sha256") or p["projection_sha256"] != m["projection_sha256"]:
            raise ValueError("daily pointer projection differs from manifest")
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
        originals = {}
        hashes = {}
        bundle = {}
        for name in ["features", "labels"]:
            content = read(uri.rsplit("/", 1)[0] + f"/contracts/{name}.json")
            hashes[name] = hashlib.sha256(content).hexdigest()
            if m.get("contract_sha256", {}).get(name, hashes[name]) != hashes[name]:
                raise ValueError("daily contract checksum mismatch")
            originals[name] = content.decode("utf-8")
            bundle[name] = json.loads(content)
        validate_contracts(m, bundle)
        bundles.append(bundle)
        manifests.append(m)
        sources.append(
            {
                "uri": uri,
                "sha256": p["manifest_sha256"],
                "date": str(day),
                "manifest": raw.decode("utf-8"),
                "contracts": originals,
                "contract_sha256": hashes,
            }
        )
    # No cross-day transaction exists in S3. Reject concurrent pointer switches
    # during resolution, then use only the pinned immutable URIs for this run.
    if any(read(uri) != body for uri, body in pointers.items()):
        raise ValueError("daily pointer changed while freezing input; retry")
    parts = [p for m in manifests for p in m["partitions"]]
    if len({p["uri"] for p in parts}) != len(parts):
        raise ValueError("duplicate daily shards")
    identity = merged_dataset_identity(manifests)
    merged = merge_contracts(bundles)
    contracts = {k: json.dumps(v, sort_keys=True).encode() for k, v in merged.items()}
    result = {
        k: manifests[0][k] for k in ["manifest_version", "dataset_id", "label_config_version"]
    }
    result.update(
        dataset_id=identity,
        daily_contracts_version="doku-daily-contracts/v1",
        feature_schema_version=merged["features"]["feature_spec_version"],
        label_definition_version=",".join(
            sorted(set().union(*[label_versions(m["label_definition_version"]) for m in manifests]))
        ),
        projection_sha256=digest(merged["features"]),
        dataset_version="frozen-" + digest(sources),
        partitions=parts,
        as_of_ms=max(m["as_of_ms"] for m in manifests),
        row_count=sum(m["row_count"] for m in manifests),
        source_manifests=sources,
        contract_sha256={k: hashlib.sha256(v).hexdigest() for k, v in contracts.items()},
    )
    result["content_sha256"] = digest(result)
    return result, contracts
