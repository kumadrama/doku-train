"""Published v5 behavior columns use the same configurable task path as immersive_click."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recommend.train.comm.datasvr.published_dataset import PublishedDataset, digest
from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.train_rerank import execute_test, execute_train

BEHAVIORS = ("like", "comment", "share", "follow", "subscribe")


def behavior_experiment(root: Path, model: str):
    config = make_experiment(root, model, rows_per_day=8)
    # Explicit fixture population: prove task masks work without a playback-finish gate.
    config["item_filter"] = "True"
    dataset = root / "dataset"
    feature_path = dataset / "contracts/features.json"
    features = json.loads(feature_path.read_text())
    features["feature_spec_version"] = "fyp-rank-features/v5"
    feature_path.write_text(json.dumps(features))
    label_path = dataset / "contracts/labels.json"
    labels = json.loads(label_path.read_text())
    labels["labels"] += list(BEHAVIORS)
    labels["binary_labels_have_valid_mask"] = False
    labels["behavior_labels"] = {"training_mask": "is_not_null_label"}
    label_path.write_text(json.dumps(labels))
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_schema_version"] = "fyp-rank-features/v5"
    for part in manifest["partitions"]:
        path = Path(urlsplit(part["uri"]).path)
        table = pq.read_table(path)
        for index, name in enumerate(BEHAVIORS):
            # Known negatives survive even without a finish event; the final row
            # represents missing upstream facts and must be masked, not filled with 0.
            values = [int((i + index) % 3 == 0) for i in range(7)] + [None]
            table = table.append_column(name, pa.array(values, type=pa.uint8()))
        table = table.set_column(
            table.schema.get_field_index("video_finish_seen"),
            "video_finish_seen",
            pa.array([0] * 8, type=pa.uint8()),
        )
        pq.write_table(table, path)
        part["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest))
    for name in BEHAVIORS:
        config["tasks"][name] = copy.deepcopy(config["tasks"]["immersive_click"])
        config["tasks"][name].update(label=name, mask=f"not is_null({name})", loss_weight="1.0")
        if model == "xgboost":
            config["selection"]["per_task_metrics"][name] = "logloss"
        else:
            config["selection"]["task_losses"][name] = 1.0
    return config


@pytest.mark.parametrize("model", ["dnn", "deepfm", "xgboost"])
def test_behavior_labels_train_and_evaluate_without_dropping_negatives(tmp_path, model):
    config = behavior_experiment(tmp_path, model)
    source = PublishedDataset(config)
    assert set(BEHAVIORS) <= source.labels
    assert not set(BEHAVIORS) & set(source.dtypes)
    rows = list(source.rows("validation", set(BEHAVIORS)))
    assert len(rows) == 8
    assert rows[-1]["like"] is None
    result = execute_train(config)
    assert result["status"] == "complete"
    metrics = execute_test(Path(result["run_dir"]))
    for index, name in enumerate(BEHAVIORS):
        assert metrics[name]["metrics"]["valid_rows"]["value"] == 7
        expected = sum((i + index) % 3 == 0 for i in range(7)) / 7
        assert metrics[name]["metrics"]["positive_rate"]["value"] == pytest.approx(expected)
        assert metrics[name]["metrics"]["logloss"]["value"] is not None
