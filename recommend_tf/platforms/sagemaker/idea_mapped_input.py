"""Resolve published mapped days to immutable upstream shards."""

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

from recommend_tf.train.comm.datasvr.daily_contracts import digest, validate_contracts
from recommend_tf.train.comm.datasvr.daily_dataset import resolve_config, window, write_window

from .publish_model import location, read


def next_ready(s3, state, source, config):
    active = state.value["active"]
    if active:
        return active["ready"]
    history = state.value["history"]
    if (
        history
        and history[-1]["stage"] != "deployed"
        and not (history[-1]["stage"] == "rolled_back" and history[-1].get("rollback_target"))
    ):
        raise ValueError("previous release failed; retry its frozen ready explicitly")
    last = state.value["high_water"] or source["bootstrap_cutoff"]
    cutoff = (date.fromisoformat(last) + timedelta(days=1)).isoformat()
    days = window(config, cutoff)
    sources = []
    for day in days["train"] + days["validation"]:
        uri = source["uri"].rstrip("/") + "/date=" + day + "/_SUCCESS"
        bucket, key = location(uri)
        response = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if not any(obj["Key"] == key for obj in response.get("Contents", [])):
            return None
        pointer = json.loads(read(s3, uri)[0])
        sources.append({"date": day, **pointer})
    return {"version": "idea-mapped-ready/v1", "cutoff": cutoff, "sources": sources}


def freeze(s3, config, ready, source, target):
    resolved = resolve_config(config, ready["cutoff"])
    split = resolved["dataset_config"]["split"]
    days = split["train_dates"] + split["validation_dates"]
    if (
        ready.get("version") != "idea-mapped-ready/v1"
        or [s["date"] for s in ready["sources"]] != days
    ):
        raise ValueError("mapped ready must pin the exact ordered window")
    parts, contracts = [], {}
    latest = 0
    for ref in ready["sources"]:
        day, uri = ref["date"], ref["manifest_uri"]
        prefix = source["uri"].rstrip("/") + "/date=" + day + "/run_id="
        if not uri.startswith(prefix) or not uri.endswith("/manifest.json"):
            raise ValueError("mapped manifest outside source day")
        body = read(s3, uri)[0]
        if hashlib.sha256(body).hexdigest() != ref["manifest_sha256"]:
            raise ValueError("mapped manifest digest mismatch")
        m = json.loads(body)
        if (
            m["dataset_id"] != "idea-rank-mapped/v1"
            or digest({k: v for k, v in m.items() if k != "content_sha256"}) != m["content_sha256"]
        ):
            raise ValueError("invalid mapped manifest")
        base = uri.rsplit("/", 1)[0] + "/"
        for name in ("features", "labels"):
            raw = read(s3, base + "contracts/" + name + ".json")[0]
            if (
                hashlib.sha256(raw).hexdigest() != m["contract_sha256"][name]
                or contracts.setdefault(name, raw) != raw
            ):
                raise ValueError("inconsistent mapped contracts")
        bundle = {k: json.loads(v) for k, v in contracts.items()}
        validate_contracts(m, bundle)
        fields = {f["name"]: f["dtype"] for f in bundle["features"]["features"]}
        if any(
            fields.get(name) != "list<int64>"
            for name in ("drama_interest_ids", "not_interested_drama_ids")
        ):
            raise ValueError("mapped publication requires upstream drama ID lists")
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        for part in m["partitions"]:
            lo, hi = [
                datetime.fromisoformat(part["event_time_" + k].replace("Z", "+00:00"))
                for k in ("from", "to")
            ]
            if not part["uri"].startswith(base) or not start <= lo < hi <= start + timedelta(
                days=1
            ):
                raise ValueError("mapped partition outside frozen run or UTC day")
        if not m["row_count"] or sum(p["rows"] for p in m["partitions"]) != m["row_count"]:
            raise ValueError("invalid mapped daily row count")
        parts.extend(m["partitions"])
        latest = max(latest, m["as_of_ms"])
    merged = {
        k: m[k]
        for k in (
            "dataset_id",
            "manifest_version",
            "feature_schema_version",
            "label_definition_version",
            "contract_sha256",
        )
    }
    merged.update(
        dataset_version="mapped-window-" + digest(ready),
        original_sources=ready["sources"],
        partitions=parts,
        as_of_ms=latest,
        row_count=sum(p["rows"] for p in parts),
    )
    merged["content_sha256"] = digest(merged)
    return write_window(resolved, ready, target, merged, contracts)
