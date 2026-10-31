import json

import pytest

from recommend_tf.train.comm.epoch_metrics import read_selected_metrics
from recommend_tf.train.comm.model_identity import model_name, recipe_sha256


def selected_run(root):
    folder = root / "training"
    (folder / "epoch-metrics").mkdir(parents=True)
    (folder / "epoch-0002").mkdir()
    (folder / "selection.json").write_text(
        json.dumps({"epoch": 2, "checkpoint": "/old/run/training/epoch-0002", "score": 0.4})
    )
    (folder / "epoch-0002/checkpoint.json").write_text(
        json.dumps({"epoch": 2, "identity": "model-digest"})
    )
    metrics = {
        "click": {"quality": {"value": None, "reason": "single_class", "rows": 7}},
        "watch": {"error": {"value": 32.0, "reason": None, "rows": 7}},
    }
    (folder / "epoch-metrics/0002.json").write_text(
        json.dumps({"epoch": 2, "validation_metrics": metrics})
    )
    return {
        "model": {
            "tasks": {
                "click": {"metrics": {"quality": {"func": "auc"}}},
                "watch": {"metrics": {"error": {"func": "mse"}}},
            }
        }
    }


def test_selected_same_epoch_and_null(tmp_path):
    config = selected_run(tmp_path)
    result = read_selected_metrics(tmp_path, config)
    assert result["selection"]["epoch"] == 2
    assert result["metrics"]["click"]["quality"]["value"] is None
    assert result["metrics"]["watch"]["error"]["value"] == 32
    path = tmp_path / "training/epoch-metrics/0002.json"
    path.write_text(path.read_text().replace('"epoch": 2', '"epoch": 3'))
    with pytest.raises(ValueError, match="evidence"):
        read_selected_metrics(tmp_path, config)


def test_identity_dates_do_not_change_recipe():
    first = {
        "model_name": "a",
        "model": {"learning_rate": 0.1},
        "dataset_config": {
            "split": {"train_from": "2026-10-01"},
            "dataset": {"manifest_uri": "s3://a"},
        },
    }
    other = dict(first, model_name="b", dataset_config={"split": {}, "dataset": {}})
    assert recipe_sha256(first) == recipe_sha256(other)
    assert recipe_sha256(first) != recipe_sha256(dict(other, new_algorithm_option=True))
    assert model_name({}) is None
    with pytest.raises(ValueError):
        model_name({}, required=True)
