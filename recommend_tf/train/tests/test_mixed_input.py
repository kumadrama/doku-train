"""Source configuration and failure isolation without external storage."""

import json
from types import SimpleNamespace

import pytest

from recommend_tf.train.comm.datasvr.mixed_input import prepare_sources, validate_sources


def source(sid=0):
    return dict(
        name=f"source_{sid}",
        source_id=sid,
        format="doku-train-dataset-manifest/v1",
        manifest_uri=f"file:///missing/{sid}",
    )


def test_duplicate_source_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        validate_sources([source(), dict(source(), name="other")])


def test_missing_sources_freeze_skip_without_successful_data(tmp_path):
    config = dict(
        data_sources=[source(), source(1)],
        dtypes={"datasvr_source": "int64"},
        dataset_config={
            "features": [],
            "feature_default_values": {},
            "execution": {"cache_dir": str(tmp_path / "cache"), "memory_limit_bytes": 1024},
        },
        histories={},
        item_filter="True",
    )
    identity, count = prepare_sources(
        config, tmp_path, SimpleNamespace(), ("train", "validation", "test"), {}
    )
    assert count == 0
    records = json.loads((tmp_path / "input/sources.json").read_text())
    assert all(r["status"] == "skipped" for r in records["sources"])
    for split in ("train", "validation", "test"):
        manifest = json.loads((tmp_path / f"prepared/{split}/complete.json").read_text())
        assert manifest == dict(identity=identity, rows=0, files=[], source_rows={})


def test_prepared_budget_counts_all_sources(tmp_path):
    from recommend_tf.train.comm.datasvr.prepared import prepare

    params = SimpleNamespace(input_features={})
    shared = tmp_path / "prepared"
    first = shared / "sources/0/train"
    prepare([{"x": [1, 2]}], first, {}, params, {}, "True", "id", 100000)
    used = sum(p.stat().st_size for p in shared.rglob("*.parquet"))
    with pytest.raises(MemoryError, match="disk budget"):
        prepare(
            [{"x": [3, 4]}],
            shared / "sources/1/train",
            {},
            params,
            {},
            "True",
            "id",
            used,
            budget_root=shared,
        )
