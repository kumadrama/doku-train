from __future__ import annotations

from recommend.train.comm.resource_report import ResourceTracker


def test_resource_tracker_records_machine_volume_and_stage_duration() -> None:
    tracker = ResourceTracker()
    with tracker.stage("feature_fit"):
        sum(range(100))
    report = tracker.report(row_count=112, batch_count=7, byte_count=4096)
    assert report.cpu_model
    assert report.cpu_count >= 1
    assert report.total_memory_bytes > 0
    assert report.available_memory_bytes > 0
    assert report.peak_rss_bytes > 0
    assert report.row_count == 112
    assert report.batch_count == 7
    assert report.byte_count == 4096
    assert report.wall_seconds >= 0.0
    assert report.rows_per_second >= 0.0
    assert report.batches_per_second >= 0.0
    assert report.bytes_per_second >= 0.0
    assert report.stage_seconds["feature_fit"] >= 0.0
