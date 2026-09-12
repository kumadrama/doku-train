from __future__ import annotations

from pathlib import Path

import pytest

from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset
from recommend.train.comm.resource_report import ResourceTracker
from recommend.train.tests.fixtures import build_training_fixture
from recommend.train.train_rerank import validate_input


@pytest.mark.performance
def test_capacity_probe_records_bounded_fixture_evidence(tmp_path: Path) -> None:
    fixture = build_training_fixture(tmp_path)
    tracker = ResourceTracker()
    with tracker.stage("decode"):
        validated = validate_input(
            manifest_uri=fixture.manifest_uri,
            config_path=fixture.config_path,
        )
        dataset = ParquetDataset(
            validated.store,
            validated.manifest,
            batch_size=validated.config.common.batch_size,
        )
        batches = tuple(dataset.iter_batches())
        row_count = sum(batch.num_rows for batch in batches)
    report = tracker.report(
        row_count=row_count,
        batch_count=len(batches),
        byte_count=sum(shard.size_bytes for shard in validated.manifest.shards),
    )
    assert report.row_count == validated.manifest.row_count
    assert report.batch_count > 0
    assert report.cpu_model
    assert report.available_memory_bytes > 0
    assert report.peak_rss_bytes < validated.config.common.execution.memory_limit_bytes
    assert report.stage_seconds["decode"] >= 0.0
