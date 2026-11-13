"""Real TF mixed-source preparation/training with synthetic manifests."""

import json
from urllib.parse import unquote, urlsplit

import pyarrow.parquet as pq
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.common_utils import eval_expr
from recommend_tf.train.models.rerank.data_utils import migrate_experiment
from recommend_tf.train.tests.experiment_fixture import make_experiment
from recommend_tf.train.train_rerank import train


def mixed_config(tmp_path):
    sources = [
        make_experiment(tmp_path / name, "deepfm", rows_per_day=3) for name in ("fyp", "idea")
    ]
    source = sources[0]
    source["max_epochs"] = 1
    manifest = json.loads((tmp_path / "fyp/dataset/manifest.json").read_text())
    schema = pq.read_schema(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    dtypes = {
        f.name: "str"
        if str(f.type) == "string"
        else "int64"
        if "int64" in str(f.type)
        else "float32"
        for f in schema
        if f.name != "impression_time"
    }
    cfg = migrate_experiment(source, dtypes, {"optimizer": "adam", "learning_rate": 0.001}, 16)
    cfg["data_sources"] = [
        dict(
            name=name,
            source_id=i,
            format="doku-train-dataset-manifest/v1",
            manifest_uri=(tmp_path / name / "dataset/manifest.json").as_uri(),
        )
        for i, name in enumerate(("fyp", "idea"))
    ]
    cfg["dtypes"]["datasvr_source"] = "int64"
    cfg["model"]["eval_tasks"] = {}
    for name, head in cfg["model"]["tasks"].items():
        head["loss_weight"] = "1.0 * (datasvr_source == 1)"
        cfg["model"]["eval_tasks"][name] = {
            "metrics": {
                metric: dict(option, filter="datasvr_source == 1")
                for metric, option in head.get("metrics", {}).items()
            }
        }
    return cfg


def test_float_times_boolean_source_gate():
    with tf.Graph().as_default(), tf.Session() as session:
        result = eval_expr("1.0 * (datasvr_source == 1)", {"datasvr_source": tf.constant([0, 1])})
        assert session.run(result).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("missing", [False, True])
def test_mixed_training_and_source_fallback(tmp_path, missing):
    cfg = mixed_config(tmp_path)
    # Six validation rows (three per source) include a partial final batch.
    cfg["batch_size"] = 4
    tasks = cfg["model"]["tasks"]
    tasks["completion"]["loss_weight"] = "0.5 * (datasvr_source == 1)"
    tasks["fast_swipe"].update(loss_weight="2.0 * (datasvr_source == 1)", loss_by_weight=True)
    if missing:
        cfg["data_sources"][0]["manifest_uri"] = (tmp_path / "missing.json").as_uri()
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    assert (tmp_path / "run/training/lineage.json").exists()
    evidence = json.loads((tmp_path / "run/input/sources.json").read_text())
    assert evidence["sources"][0]["status"] == ("skipped" if missing else "ready")
    resolved = json.loads((tmp_path / "run/resolved-config.json").read_text())
    assert resolved["model"] == cfg["model"]
    record = result[0]
    metrics = record["validation_metrics"]
    all_rows = 3 if missing else 6
    for name, expected_scale in [("completion", 0.5 * 3 / all_rows), ("fast_swipe", 2.0)]:
        logloss = metrics[name]["logloss"]
        assert logloss["rows"] == 3
        assert record["validation_loss"][name] == pytest.approx(
            expected_scale * logloss["value"], rel=1e-5
        )


def test_old_sum_recipes_rejected_before_mixed_preparation_or_training(tmp_path):
    from recommend_tf.train.comm.training_pipeline import run
    from recommend_tf.train.experiment import execute

    cfg = mixed_config(tmp_path)
    cfg["model"]["tasks"]["completion"]["loss_reduction"] = "sum"
    with pytest.raises(ValueError, match="Unsupported layer settings: .*loss_reduction"):
        execute(cfg, tmp_path / "manifest-run")
    assert not (tmp_path / "manifest-run/prepared").exists()

    cfg["prepared_identity"] = "unused"
    with pytest.raises(ValueError, match="Unsupported layer settings: .*loss_reduction"):
        run(cfg, tmp_path / "missing-prepared", tmp_path / "prepared-run")
    assert not (tmp_path / "prepared-run/epoch-0001").exists()


def test_all_sources_missing_skip(tmp_path):
    cfg = mixed_config(tmp_path)
    for entry in cfg["data_sources"]:
        entry["manifest_uri"] = (tmp_path / (entry["name"] + "-missing.json")).as_uri()
    result = train(cfg, tmp_path / "run")
    assert result["status"] == "skipped"
    assert not (tmp_path / "run/training/lineage.json").exists()


def test_empty_validation_and_test_keep_final_checkpoint(tmp_path):
    cfg = mixed_config(tmp_path)
    split = cfg["dataset_config"]["split"]
    split.update(
        validation_from="2030-01-01",
        validation_to="2030-01-01",
        test_from="2030-01-02",
        test_to="2030-01-02",
    )
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    assert all(
        metric["value"] is None
        for task in result[0]["validation_metrics"].values()
        for metric in task.values()
    )
    assert (tmp_path / "run/training/epoch-0001/model.index").exists()
    from recommend_tf.train.comm.training_pipeline import evaluate_checkpoint

    runtime = json.loads((tmp_path / "run/resolved-config.json").read_text())
    measured = evaluate_checkpoint(
        runtime,
        tmp_path / "run/prepared",
        tmp_path / "run/training/epoch-0001",
        tmp_path / "evaluation",
        "test",
    )
    assert all(
        metric["value"] is None for task in measured["metrics"].values() for metric in task.values()
    )


def test_corrupt_source_does_not_leak_partial_rows(tmp_path):
    cfg = mixed_config(tmp_path)
    manifest = json.loads((tmp_path / "fyp/dataset/manifest.json").read_text())
    path = unquote(urlsplit(manifest["partitions"][-1]["uri"]).path)
    from pathlib import Path

    Path(path).write_bytes(b"corrupt")
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    prepared = json.loads((tmp_path / "run/prepared/train/complete.json").read_text())
    assert set(prepared["source_rows"]) == {"1"}


def test_frozen_mixed_replay_and_checkpoint_resume(tmp_path):
    from pathlib import Path

    from recommend_tf.train.comm.training_input import train as input_train
    from recommend_tf.train.comm.training_pipeline import run
    from recommend_tf.train.models.rerank.model_params import parameters

    cfg = mixed_config(tmp_path)
    cfg["max_epochs"] = 2
    baseline = train(cfg, tmp_path / "baseline")
    # Frozen manifests/contracts, not the mutable source metadata, govern replay.
    for source in cfg["data_sources"]:
        Path(unquote(urlsplit(source["manifest_uri"]).path)).write_text("invalid replacement")
    replay = input_train(
        cfg,
        tmp_path / "replay",
        model_parameters=parameters,
        frozen_input=tmp_path / "baseline/input",
    )
    for actual, expected in zip(replay, baseline):
        assert actual["train_loss"] == expected["train_loss"]
        assert actual["validation_metrics"] == expected["validation_metrics"]
    runtime = json.loads((tmp_path / "baseline/resolved-config.json").read_text())
    resumed = run(
        runtime,
        tmp_path / "baseline/prepared",
        tmp_path / "resumed",
        resume=tmp_path / "baseline/training/epoch-0001",
    )
    assert len(resumed) == 1
    assert resumed[0]["train_loss"] == baseline[1]["train_loss"]
    assert resumed[0]["validation_metrics"] == baseline[1]["validation_metrics"]


def test_placement_loss_gate_excludes_missing_and_other_sources():
    from recommend_tf.train.comm.nn_utils import get_loss

    with tf.Graph().as_default(), tf.Session() as session:
        fields = {
            "datasvr_source": tf.constant([1, 1, 1, 0]),
            "placement": tf.constant(["WATCH_NOW", "REGULAR", "", "WATCH_NOW"]),
        }
        head = dict(
            loss_weight="1.0 * ((datasvr_source == 1) and (placement == 'WATCH_NOW'))",
            loss_type="logloss",
            predict_fn="sigmoid",
        )
        loss, weights = get_loss(head, tf.zeros([4]), tf.ones([4]), {"item": fields})
        actual_loss, actual_weights = session.run([loss, weights])
        assert actual_weights.tolist() == [1.0, 0.0, 0.0, 0.0]
        import math

        assert actual_loss == pytest.approx(math.log(2) / 4)
