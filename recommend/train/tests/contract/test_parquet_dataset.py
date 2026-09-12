from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recommend.train.comm.datasvr.dataset_manifest import (
    DatasetManifest,
    calculate_manifest_sha256,
)
from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset


def build_manifest(tmp_path: Path, *, rows_per_day: int = 3) -> DatasetManifest:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    shards: list[dict[str, object]] = []
    for offset in range(2):
        event_time = start + timedelta(days=offset)
        table = pa.table(
            {
                "event_time_ms": [int(event_time.timestamp() * 1000)] * rows_per_day,
                "watch_duration_ms": [94, 95, 96][:rows_per_day],
                "content_duration_ms": [100] * rows_per_day,
            }
        )
        path = tmp_path / f"day={event_time.date().isoformat()}" / "part.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)
        body = path.read_bytes()
        shards.append(
            {
                "uri": path.as_uri(),
                "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "row_count": rows_per_day,
                "event_date": event_time.date().isoformat(),
                "min_event_time_ms": int(event_time.timestamp() * 1000),
                "max_event_time_ms": int(event_time.timestamp() * 1000),
            }
        )
    empty_digest = hashlib.sha256(b"{}").hexdigest()
    payload: dict[str, object] = {
        "manifest_version": "1",
        "dataset_id": "fixture",
        "dataset_version": "v1",
        "feature_schema_version": "features-v1",
        "feature_schema": {
            "uri": (tmp_path / "features.json").as_uri(),
            "size_bytes": 2,
            "sha256": empty_digest,
        },
        "label_definition_version": "labels-v1",
        "label_definition": {
            "uri": (tmp_path / "labels.json").as_uri(),
            "size_bytes": 2,
            "sha256": empty_digest,
        },
        "as_of_ms": int((start + timedelta(days=4)).timestamp() * 1000),
        "label_maturity_hours": 24,
        "shards": shards,
        "row_count": rows_per_day * 2,
    }
    payload["content_sha256"] = calculate_manifest_sha256(payload)
    return DatasetManifest.model_validate(payload)


def test_reader_streams_ordered_bounded_batches(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    dataset = ParquetDataset(LocalStorage(tmp_path), manifest, batch_size=2)
    batches = list(dataset.iter_batches())
    assert [batch.num_rows for batch in batches] == [2, 1, 2, 1]
    assert sum(batch.num_rows for batch in batches) == manifest.row_count
    assert batches[0].column("watch_duration_ms").to_pylist() == [94, 95]


def test_reader_fails_on_shard_checksum_mismatch(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    first_path = Path(unquote(urlsplit(manifest.shards[0].uri).path))
    first_path.write_bytes(b"corrupt")
    dataset = ParquetDataset(LocalStorage(tmp_path), manifest, batch_size=2)
    with pytest.raises(ValueError, match="DATA_INTEGRITY_MISMATCH"):
        list(dataset.iter_batches())
