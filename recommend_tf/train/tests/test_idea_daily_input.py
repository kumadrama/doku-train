import pyarrow as pa
import pyarrow.parquet as pq

from recommend_tf.platforms.sagemaker.idea_mapped_input import dates, project


def test_growing_window_ends_on_validation_date():
    first = dates("2026-11-02", "2026-11-12")
    full = dates("2026-11-02", "2026-12-12")
    assert (first[0], first[-1], len(first)) == ("2026-11-02", "2026-11-12", 11)
    assert (full[0], full[-1], len(full)) == ("2026-11-12", "2026-12-12", 31)


def test_projection_keeps_key_order_empty_and_null(tmp_path):
    values = [[(9, 1), (2, 5)], [], None]
    history = pa.array(values, type=pa.map_(pa.int64(), pa.int64()))
    source, target = tmp_path / "source.parquet", tmp_path / "projection.parquet"
    pq.write_table(
        pa.table({"drama_interests_v1": history, "not_interested_dramas": history}), source
    )
    assert project(source, target) == 3
    projected = pq.read_table(target)
    for field in ("drama_interest_ids", "not_interested_drama_ids"):
        assert projected[field].to_pylist() == [[9, 2], [], None]
