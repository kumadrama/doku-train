from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np
import pytest

from recommend.train.comm.batch_feature_plan import BatchFeaturePlan
from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.tests.experiment_fixture import make_experiment


class CountingDataset(PublishedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = Counter()

    def iter_batches(self, split, columns):
        self.scans[split] += 1
        yield from super().iter_batches(split, columns)

    def rows(self, split, columns):
        self.scans[split] += 1
        yield from super().rows(split, columns)


@pytest.mark.parametrize("workers", [0, 2])
def test_full_preparation_parity_and_scan_counts(tmp_path, workers):
    config = make_experiment(tmp_path / "input", "deepfm", rows_per_day=19)
    config["batch_size"] = 7
    config["execution"].update(reader_workers=workers, s3_transfer_threads=2, prefetch_batches=2)
    data = CountingDataset(config)
    old = FeaturePlan(config, data.dtypes)
    old.fit_dataset(data)
    old_saved = json.dumps(old.state(), sort_keys=True)
    expected = {
        s: old.prepare_dataset(data, s, tmp_path / ("old-" + s))
        for s in ["train", "validation", "test"]
    }
    assert data.scans == {"train": 3, "validation": 2, "test": 2}
    data.scans.clear()
    new = BatchFeaturePlan(config, data.dtypes)
    new.fit_dataset(data)
    assert json.dumps(new.state(), sort_keys=True) == old_saved
    for split in expected:
        actual = new.prepare_dataset(data, split, tmp_path / ("new-" + split))
        assert list(actual) == list(expected[split])
        for key in actual:
            a, b = actual[key], expected[split][key]
            assert a.shape == b.shape and a.dtype == b.dtype, key
            assert a.tobytes() == b.tobytes(), key
    assert data.scans == {"train": 2, "validation": 2, "test": 2}
    assert new.state() == old.state()


def test_weighted_history_and_mask_parity(tmp_path):
    config = make_experiment(tmp_path, "deepfm", rows_per_day=8)
    data = PublishedDataset(config)
    old = FeaturePlan(config, data.dtypes)
    new = BatchFeaturePlan(config, data.dtypes)
    raw = list(data.rows("train", old.columns))[:6]
    raw[0]["video_finish_seen"] = 0
    raw[0]["u_drama_interests__weight"] = [float("nan")]
    raw[1]["u_drama_interests__id"] = [2, 1, 2, 3]
    raw[1]["u_drama_interests__weight"] = [1e300, 1e300, 1e-300, 0.0]
    raw[2]["u_drama_interests__id"] = list(range(150))
    raw[2]["u_drama_interests__weight"] = [1.0] * 150
    raw[3]["u_drama_interests__id"] = []
    raw[3]["u_drama_interests__weight"] = []
    raw[4]["watch_duration_ms"] = None
    rows = [r for x in raw if (r := old.row(x)) is not None]
    cols = new.transform_batch({k: [r[k] for r in raw] for k in raw[0]})
    assert [{k: v[i] for k, v in cols.items()} for i in range(len(rows))] == rows
    old.fit(iter(raw))
    new.restore(copy.deepcopy(old.state()))
    encoded = new.encode_batch(cols)
    for k, v in encoded.items():
        expected = np.stack([old.encode(row)[k] for row in rows])
        assert expected.tobytes() == v.tobytes(), k


def test_empty_filtered_and_budget_failure(tmp_path):
    config = make_experiment(tmp_path / "input", "deepfm", rows_per_day=8)
    config["item_filter"] = "False"
    data = PublishedDataset(config)
    plan = BatchFeaturePlan(config, data.dtypes)
    plan.fit_dataset(data)
    with pytest.raises(ValueError, match="no samples"):
        plan.prepare_dataset(data, "train", tmp_path / "empty")
    config["item_filter"] = "True"
    data = PublishedDataset(config)
    plan = BatchFeaturePlan(config, data.dtypes)
    plan.fit_dataset(data)
    data.config["execution"]["cache_limit_bytes"] = 1
    with pytest.raises(ValueError, match="disk budget"):
        plan.prepare_dataset(data, "train", tmp_path / "full")


def test_widths_restore_longer_validation_and_history_scalar(tmp_path):
    config = make_experiment(tmp_path / "input", "deepfm", rows_per_day=8)
    config["history_scalars"] = {"hist_count": {"func": "count", "history": "hist_drama_skip"}}
    data = PublishedDataset(config)
    raw = list(
        data.rows(
            "train",
            set(config["features"])
            | {
                "video_finish_seen",
                "watch_duration_ms",
                "completion",
                "non_fast_swipe",
                "immersive_click",
            },
        )
    )[:3]
    raw[0]["i_genre_v2_ids"] = [0]
    raw[1]["i_genre_v2_ids"] = [0, 1]

    class Data:
        def __init__(self):
            self.config = config
            self.calls = Counter()

        def partitions(self, split):
            return [{"rows": len(raw)}]

        def rows(self, split, columns):
            self.calls[split] += 1
            rows = copy.deepcopy(raw)
            if split != "train":
                rows[0]["i_genre_v2_ids"] = [0, 1, 2, 3, 4]
            yield from rows

        def iter_batches(self, split, columns):
            rows = list(self.rows(split, columns))
            yield {k: [r[k] for r in rows] for k in rows[0]}

    dataset = Data()
    old = FeaturePlan(config, data.dtypes)
    new = BatchFeaturePlan(config, data.dtypes)
    old.fit_dataset(dataset)
    new.fit_dataset(dataset)
    for split in ["train", "validation"]:
        a = old.prepare_dataset(dataset, split, tmp_path / ("old-" + split))
        b = new.prepare_dataset(dataset, split, tmp_path / ("new-" + split))
        for key in a:
            assert a[key].tobytes() == b[key].tobytes()
    new.restore(copy.deepcopy(old.state()))
    dataset.calls.clear()
    new.prepare_dataset(dataset, "train", tmp_path / "restored")
    assert dataset.calls["train"] == 2  # Restoring state is not proof of a fit scan.


def test_history_mismatch_and_filtered_invalid_rows(tmp_path):
    config = make_experiment(tmp_path, "deepfm", rows_per_day=8)
    data = PublishedDataset(config)
    old = FeaturePlan(config, data.dtypes)
    new = BatchFeaturePlan(config, data.dtypes)
    row = next(data.rows("train", old.columns))
    row["u_drama_interests__weight"] = [1.0]
    for plan in [old, new]:
        with pytest.raises(ValueError, match="mismatch"):
            if isinstance(plan, BatchFeaturePlan):
                plan.transform_batch({k: [v] for k, v in row.items()})
            else:
                plan.row(row)
    row["video_finish_seen"] = 0
    assert old.row(row) is None
    assert len(next(iter(new.transform_batch({k: [v] for k, v in row.items()}).values()))) == 0


@pytest.mark.parametrize("fail", [False, True])
def test_scalar_refit_invalidates_scan_proof(tmp_path, fail):
    config = make_experiment(tmp_path / "input", "deepfm", rows_per_day=8)
    dataset = CountingDataset(config)
    plan = BatchFeaturePlan(config, dataset.dtypes)
    plan.fit_dataset(dataset)
    assert plan.fitted_dataset is dataset

    def rows():
        if fail:
            raise ValueError("replacement fit failed")
        yield from dataset.rows("train", plan.columns)

    if fail:
        with pytest.raises(ValueError, match="replacement fit failed"):
            plan.fit(rows())
    else:
        plan.fit(rows())
    assert plan.fitted_dataset is None
    assert plan.widths == {}
    if not fail:
        dataset.scans.clear()
        plan.prepare_dataset(dataset, "train", tmp_path / "arrays")
        assert dataset.scans["train"] == 2
