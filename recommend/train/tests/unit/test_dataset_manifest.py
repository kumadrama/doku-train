from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from recommend.train.comm.datasvr.dataset_manifest import (
    DatasetManifest,
    calculate_manifest_sha256,
)


def manifest_payload() -> dict[str, object]:
    body: dict[str, object] = {
        "manifest_version": "1",
        "dataset_id": "rerank-exposures",
        "dataset_version": "2026-09-01",
        "feature_schema_version": "features-v1",
        "feature_schema": {
            "uri": "s3://doku-data/contracts/features-v1.json",
            "size_bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "label_definition_version": "labels-v1",
        "label_definition": {
            "uri": "s3://doku-data/contracts/labels-v1.json",
            "size_bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "as_of_ms": 1_800_000_000_000,
        "label_maturity_hours": 24,
        "shards": [
            {
                "uri": "s3://doku-data/training/day=2026-09-01/part.parquet",
                "size_bytes": 10,
                "sha256": hashlib.sha256(b"parquet").hexdigest(),
                "row_count": 2,
                "event_date": "2026-09-01",
                "min_event_time_ms": 1_799_000_000_000,
                "max_event_time_ms": 1_799_000_001_000,
            }
        ],
        "row_count": 2,
    }
    body["content_sha256"] = calculate_manifest_sha256(body)
    return body


def test_manifest_accepts_exact_canonical_digest() -> None:
    manifest = DatasetManifest.model_validate(manifest_payload())
    assert manifest.row_count == 2
    assert manifest.content_sha256 == calculate_manifest_sha256(manifest.model_dump(mode="json"))


def test_manifest_rejects_total_row_mismatch() -> None:
    payload = manifest_payload()
    payload["row_count"] = 3
    payload["content_sha256"] = calculate_manifest_sha256(payload)
    with pytest.raises(ValidationError, match="row_count"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_duplicate_shard_uri() -> None:
    payload = manifest_payload()
    payload["shards"] = [*payload["shards"], payload["shards"][0]]  # type: ignore[index]
    payload["row_count"] = 4
    payload["content_sha256"] = calculate_manifest_sha256(payload)
    with pytest.raises(ValidationError, match="duplicate shard URI"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_digest_mismatch() -> None:
    payload = manifest_payload()
    payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256"):
        DatasetManifest.model_validate(payload)


def test_manifest_json_is_order_independent() -> None:
    payload = manifest_payload()
    reversed_payload = dict(reversed(payload.items()))
    assert calculate_manifest_sha256(payload) == calculate_manifest_sha256(reversed_payload)
    assert json.loads(DatasetManifest.model_validate(payload).canonical_json())["dataset_id"] == (
        "rerank-exposures"
    )
