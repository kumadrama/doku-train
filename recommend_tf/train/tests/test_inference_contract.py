import copy
import json

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.inference.artifact_loader import load
from recommend_tf.inference.request_adapter import prepare
from recommend_tf.inference.response_adapter import format_response
from recommend_tf.train.comm.datasvr.prepared import organize, tensor_batch
from recommend_tf.train.experiment import execute
from recommend_tf.train.export_model import execute as export
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_daily_publication import fixture

TASKS = {
    "completion": "probability",
    "fast_swipe": "probability",
    "immersive_click": "probability",
    "watch_seconds": "seconds",
}


def exported_fixture(tmp_path):
    config, _ = fixture(tmp_path)
    execute(config, tmp_path / "run")
    export(tmp_path / "run", tmp_path / "package", outputs=TASKS)
    artifact = load(tmp_path / "package")
    cfg = artifact["config"]
    history = {
        f
        for g in cfg["input_features"].values()
        if g.get("history_pooling")
        for f in g["features"] + ([g["weight_feature"]] if "weight_feature" in g else [])
    }
    rows = []
    for size in (0, 1, 7):
        row = {}
        for name, dtype in cfg["dtypes"].items():
            value = "oov" if dtype == "str" else 999
            row[name] = [value] * size if name in history else value
        rows.append(row)
    return config, artifact, {"schema_version": "doku-tf-request/v1", "instances": rows}


def test_raw_request_matches_training_tensors_and_savedmodel(tmp_path):
    config, artifact, request = exported_fixture(tmp_path)
    payload, size = prepare(request, artifact, max_batch=10, max_history=100)
    params = parameters(config["model"])
    # Independent training path provides the oracle; inference imports no train code.
    raw = []
    for row in request["instances"]:
        complete = {k: "" if v == "str" else 0 for k, v in config["dtypes"].items()}
        complete.update(row)
        raw.append(
            organize(
                complete,
                config["dataset_config"]["feature_default_values"],
                params.input_features,
                config["histories"],
            )
        )
    expected = tensor_batch(raw, params, config["dtypes"])
    for key, spec in artifact["signature"]["inputs"].items():
        np.testing.assert_array_equal(
            payload["inputs"][key], expected[spec["source_group"]][spec["source_field"]]
        )
    with tf.Graph().as_default(), tf.Session() as session:
        meta = tf.saved_model.loader.load(session, ["serve"], str(tmp_path / "package/saved_model"))
        signature = meta.signature_def["serving_default"]
        result = session.run(
            {k: session.graph.get_tensor_by_name(v.name) for k, v in signature.outputs.items()},
            {
                session.graph.get_tensor_by_name(signature.inputs[k].name): v
                for k, v in payload["inputs"].items()
            },
        )
    response = format_response(
        {"outputs": {k: v.tolist() for k, v in result.items()}},
        artifact["version"],
        size,
        artifact["signature"]["outputs"],
    )
    assert len(response["predictions"]) == 3 and response["model_version"] == artifact["version"]
    bad = copy.deepcopy(request)
    bad["instances"][0]["unapproved_label"] = 1
    with pytest.raises(ValueError, match="unknown"):
        prepare(bad, artifact, max_batch=10, max_history=100)
    (tmp_path / "package/feature-state.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        load(tmp_path / "package")


def test_response_rejects_nonfinite_and_batch_mismatch():
    with pytest.raises(ValueError):
        format_response({"outputs": {k: [float("nan")] for k in TASKS}}, "v1", 1, TASKS)
    with pytest.raises(ValueError):
        format_response({"outputs": {k: [1.0] for k in TASKS}}, "v1", 2, TASKS)


def test_weighted_history_defaults_and_order():
    artifact = {
        "config": {
            "required_raw_fields": ["ids", "weights", "score"],
            "dtypes": {"ids": "int64", "weights": "float32", "score": "float32"},
            "defaults": {"ids": [], "weights": [], "score": -1},
            "input_features": {
                "hist": {
                    "from": "history",
                    "features": ["ids"],
                    "weight_feature": "weights",
                    "history_pooling": "sum",
                }
            },
            "histories": {
                "hist": {
                    "max_hist_size": 2,
                    "truncate": "top_weight",
                    "duplicates": "sum_weights",
                    "tie_break": "raw_id_ascending",
                }
            },
        },
        "signature": {
            "inputs": {
                "history__ids": {
                    "source_group": "history",
                    "source_field": "ids",
                    "dtype": "int64",
                    "shape": [None, None],
                },
                "history__length": {
                    "source_group": "history",
                    "source_field": "length",
                    "dtype": "int64",
                    "shape": [None],
                },
                "item__score": {
                    "source_group": "item",
                    "source_field": "score",
                    "dtype": "float32",
                    "shape": [None],
                },
            }
        },
    }
    payload = {
        "schema_version": "doku-tf-request/v1",
        "instances": [
            {},
            {"ids": [3, 2, 3, 1], "weights": [1.0, 2.0, 1.0, 0.0]},
        ],
    }
    tensors, _ = prepare(payload, artifact, max_batch=4, max_history=10)
    assert tensors["inputs"] == {
        "history__ids": [[0, 0], [2, 3]],
        "history__length": [0, 2],
        "item__score": [-1, -1],
    }
    for bad in ({"ids": [1]}, {"ids": [1], "weights": [-1.0]}, {"score": float("nan")}):
        with pytest.raises(ValueError):
            prepare(
                {"schema_version": "doku-tf-request/v1", "instances": [bad]},
                artifact,
                max_batch=4,
                max_history=10,
            )


def test_export_only_selected_task_preserves_training_heads(tmp_path):
    config, _ = fixture(tmp_path)
    config["export_tasks"] = ["completion"]
    execute(config, tmp_path / "run")
    export(tmp_path / "run", tmp_path / "package")
    artifact = load(tmp_path / "package")
    assert artifact["signature"]["output_units"] == {"completion": "probability"}
    assert set(artifact["signature"]["outputs"]) == {"completion"}
    runtime = json.loads((tmp_path / "run/resolved-config.json").read_text())
    assert runtime["model"]["tasks"] == config["model"]["tasks"]
    assert len(runtime["model"]["tasks"]) == 7
    checkpoint = tf.train.list_variables(str(tmp_path / "run/training/epoch-0001/model"))
    for task in runtime["model"]["tasks"]:
        assert any(f"/{task}/" in name for name, _ in checkpoint), task
    assert "export_tasks" not in runtime
    with tf.Graph().as_default(), tf.Session() as session:
        meta = tf.saved_model.loader.load(session, ["serve"], str(tmp_path / "package/saved_model"))
        assert set(meta.signature_def["serving_default"].outputs) == {"completion"}
    assert format_response({"outputs": [0.25]}, "v1", 1, ["completion"])["predictions"] == [
        {"completion": 0.25}
    ]
    with pytest.raises(ValueError, match="tasks differ"):
        format_response({"outputs": {"aux": [0.25]}}, "v1", 1, ["completion"])


@pytest.mark.parametrize("names", [[], ["a", "a"], ["typo"], "a", None])
def test_invalid_export_task_selection(names):
    from recommend_tf.train.comm.export_tasks import export_outputs

    with pytest.raises(ValueError, match="export_tasks"):
        export_outputs({"model": {"tasks": {"a": {}}}, "export_tasks": names}, {})
