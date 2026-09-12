from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from recommend.train.comm.datasvr.dataset_manifest import (
    DatasetManifest,
    calculate_manifest_sha256,
)
from recommend.train.models.rerank.data_utils import build_daily_split, completion_labels


def daily_manifest(*, days: int = 28, missing_offset: int | None = None) -> DatasetManifest:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    shards: list[dict[str, object]] = []
    for offset in range(days):
        actual_offset = (
            offset + 1 if missing_offset is not None and offset >= missing_offset else offset
        )
        event_time = start + timedelta(days=actual_offset)
        shards.append(
            {
                "uri": f"s3://doku/training/{event_time.date().isoformat()}.parquet",
                "size_bytes": 1,
                "sha256": hashlib.sha256(str(offset).encode()).hexdigest(),
                "row_count": 1,
                "event_date": event_time.date().isoformat(),
                "min_event_time_ms": int(event_time.timestamp() * 1000),
                "max_event_time_ms": int(event_time.timestamp() * 1000),
            }
        )
    empty_digest = hashlib.sha256(b"{}").hexdigest()
    last_event_time = start + timedelta(days=days - 1 + (1 if missing_offset is not None else 0))
    payload: dict[str, object] = {
        "manifest_version": "1",
        "dataset_id": "fixture",
        "dataset_version": "v1",
        "feature_schema_version": "features-v1",
        "feature_schema": {
            "uri": "s3://doku/training/features.json",
            "size_bytes": 2,
            "sha256": empty_digest,
        },
        "label_definition_version": "labels-v1",
        "label_definition": {
            "uri": "s3://doku/training/labels.json",
            "size_bytes": 2,
            "sha256": empty_digest,
        },
        "as_of_ms": int((last_event_time + timedelta(hours=25)).timestamp() * 1000),
        "label_maturity_hours": 24,
        "shards": shards,
        "row_count": len(shards),
    }
    payload["content_sha256"] = calculate_manifest_sha256(payload)
    return DatasetManifest.model_validate(payload)


def test_daily_split_is_27_train_1_validation_and_28_refit() -> None:
    split = build_daily_split(daily_manifest())
    assert len(split.train) == 27
    assert len(split.validation) == 1
    assert len(split.refit) == 28
    assert split.validation[0].event_date.isoformat() == "2026-01-28"
    assert split.refit == (*split.train, *split.validation)


def test_daily_split_allows_multiple_deterministic_shards_per_day() -> None:
    manifest = daily_manifest()
    payload = manifest.model_dump(mode="json")
    first = dict(payload["shards"][0])
    first["uri"] = "s3://doku/training/2026-01-01-part-2.parquet"
    first["sha256"] = hashlib.sha256(b"part-2").hexdigest()
    payload["shards"] = [*payload["shards"], first]
    payload["row_count"] = manifest.row_count + 1
    payload["content_sha256"] = calculate_manifest_sha256(payload)

    split = build_daily_split(DatasetManifest.model_validate(payload))
    assert len(split.train) == 28
    assert len(split.validation) == 1
    assert len(split.refit) == 29
    assert [shard.uri for shard in split.train[:2]] == sorted(
        [manifest.shards[0].uri, str(first["uri"])]
    )


def test_daily_split_rejects_missing_or_immature_day() -> None:
    with pytest.raises(ValueError, match="SPLIT_INVALID"):
        build_daily_split(daily_manifest(missing_offset=10))
    manifest = daily_manifest()
    immature = manifest.model_copy(
        update={"as_of_ms": manifest.shards[-1].max_event_time_ms + 23 * 3_600_000}
    )
    with pytest.raises(ValueError, match="LABEL_NOT_MATURE"):
        build_daily_split(immature)


def test_completion_labels_include_95_percent_boundary_and_mask_invalid_duration() -> None:
    labels, valid = completion_labels(
        np.array([94.0, 95.0, 96.0, 0.0, np.nan]),
        np.array([100.0, 100.0, 100.0, 0.0, 100.0]),
    )
    assert labels.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert valid.tolist() == [True, True, True, False, False]
