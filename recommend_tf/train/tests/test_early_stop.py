import copy
import json
import re
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recommend_tf.train.comm.datasvr.prepared import prepare
from recommend_tf.train.comm.early_stop import EarlyStop, validate_early_stop
from recommend_tf.train.comm.eval.metrics import Metrics
from recommend_tf.train.comm.training_pipeline import run
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_learning_rate import assert_same_checkpoint, logical
from recommend_tf.train.tests.test_migration import config as model_config


def options(metric="auc", patient=2, **extra):
    return {
        "patient": patient,
        "metric": metric,
        "weight": {"a": 0.5, "b": 0.25, "c": 0.25},
        **extra,
    }


def configured(early):
    return {"model": model_config(), "early_stop": early}


def observed(a, b=None, c=None, metric="auc"):
    return {
        task: {metric: {"value": value, "reason": "single_class" if value is None else None}}
        for task, value in zip(("a", "b", "c"), (a, a if b is None else b, a if c is None else c))
    }


def test_optional_config_and_normalized_weights():
    assert validate_early_stop({}) is None
    cfg = configured(options(weight={"a": 2, "b": 1, "c": 1, "watch": 0}))
    spec = validate_early_stop(cfg)
    assert spec["weight"] == {"a": 0.5, "b": 0.25, "c": 0.25}
    stop = EarlyStop(spec)
    assert stop.observe(observed(0.8, 0.6, 0.4), 1, "e1")["score"] == pytest.approx(0.65)
    stop.observe(observed(0.79, 0.8, 0.8), 2, "e2")
    assert stop.best["epoch"] == 2  # Composite improves while click falls.
    assert cfg["early_stop"]["weight"]["a"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        False,
        options(patient=0),
        options(patient=-1),
        options(patient=True),
        options(patient=1.5),
        options(metric="auc / logloss"),
        options(metric="mse"),
        options(weight={}),
        options(weight={"missing": 1}),
        options(weight={"watch": 1}),
        options(weight={"a": -1}),
        options(weight={"a": 0}),
        options(weight={"a": True}),
        options(weight={"a": float("nan")}),
        options(weight={"a": float("inf")}),
        options(min_delta=-1),
        options(min_delta=float("nan")),
        options(min_delta=True),
        options(patience=8),
        options(weight=[1]),
    ],
)
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        validate_early_stop(configured(bad))


@pytest.mark.parametrize(
    "metric,values", [("auc", [0.7, 0.8, 0.8, 0.79]), ("logloss", [0.7, 0.6, 0.6, 0.61])]
)
def test_direction_ties_and_exact_stop_boundary(metric, values):
    stop = EarlyStop(validate_early_stop(configured(options(metric))))
    for epoch, value in enumerate(values, 1):
        result = stop.observe(observed(value, metric=metric), epoch, f"e{epoch}")
        assert result["stopped"] == (epoch == 4)
    assert stop.best["epoch"] == 2 and stop.state["wait"] == 2
    with pytest.raises(ValueError, match="already stopped"):
        stop.observe(observed(values[-1], metric=metric), 5, "e5")


def test_min_delta_keeps_true_best_and_resume_counter():
    stop = EarlyStop(validate_early_stop(configured(options(min_delta=0.01))))
    stop.observe(observed(0.7), 1, "e1")
    stop.observe(observed(0.705), 2, "e2")
    assert stop.best["epoch"] == 2 and stop.state["wait"] == 1
    resumed = EarlyStop(stop.options, json.loads(json.dumps(stop.state)))
    result = resumed.observe(observed(0.709), 3, "e3")
    assert result["stopped"] and resumed.best["epoch"] == 3
    assert stop.state["wait"] == 1  # Restored state is independent.


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_unavailable_metric_freezes_counter_without_reweighting(value):
    stop = EarlyStop(validate_early_stop(configured(options())))
    stop.observe(observed(0.8), 1, "e1")
    stop.observe(observed(0.7), 2, "e2")
    metrics = observed(0.95)
    metrics["b"]["auc"] = {"value": value, "reason": "single_class"}
    result = stop.observe(metrics, 3, "e3")
    assert result["score"] is None and result["wait"] == 1 and not stop.stopped
    assert result["unavailable"] == {"b": "single_class"}
    assert stop.best["epoch"] == 1
    assert stop.observe(observed(0.7), 4, "e4")["stopped"]


def runtime(tmp_path, max_epochs=7):
    model = model_config()
    model["tasks"] = {k: v for k, v in model["tasks"].items() if k in ("a", "b", "c")}
    model.update(
        optimizer="adamw",
        learning_rate=0.001,
        optimizer_args={"weight_decay": 0.0001},
        learning_rate_schedule={
            "type": "piecewise_constant",
            "points": [{"epoch": 1, "learning_rate": 0.001}, {"epoch": 3, "learning_rate": 0.0005}],
        },
    )
    examples = [
        dict(id=i % 3, history=[0, 1][: i % 3], value=float(i % 3), click=i % 2, scene=i % 2)
        for i in range(8)
    ]
    for split in ("train", "validation", "test"):
        prepare(
            [examples],
            tmp_path / "input" / split,
            {},
            parameters(model),
            {},
            "True",
            "fixture",
            1000000,
        )
    return dict(
        model=model,
        prepared_identity="fixture",
        dtypes=dict(id="int64", history="int64", value="float32", click="int64", scene="int64"),
        batch_size=4,
        seed=42,
        max_epochs=max_epochs,
        selection={"mode": "min", "task_losses": {"a": 1.0}},
        shuffle_buffer_size=4,
        intra_op_threads=1,
        inter_op_threads=1,
    )


def override_validation(monkeypatch, values, metric="auc"):
    original = Metrics.finish

    def finish(self):
        measured = original(self)
        path = self.db.execute("PRAGMA database_list").fetchone()[2]
        match = re.search(r"metrics-(\d+)\.sqlite$", path)
        if match:
            value = values[int(match.group(1)) - 1]
            for task in measured:
                if metric in measured[task]:
                    measured[task][metric].update(
                        value=value, reason="single_class" if value is None else None
                    )
        return measured

    monkeypatch.setattr(Metrics, "finish", finish)


def test_train_early_stop_disabled_and_resume(tmp_path, monkeypatch):
    cfg = runtime(tmp_path)
    override_validation(monkeypatch, [0.7, 0.8, 0.79, None, 0.78, 0.77, 0.76])
    full = run(cfg, tmp_path / "input", tmp_path / "full")
    assert len(full) == 7 and "early_stop" not in full[0]
    assert not (tmp_path / "full/early-stop.json").exists()
    cfg["early_stop"] = options()
    seen = []
    early = run(
        cfg, tmp_path / "input", tmp_path / "early", on_epoch=lambda r: seen.append(r["epoch"])
    )
    assert seen == [1, 2, 3, 4, 5] and len(early) == 5
    assert early[-1]["early_stop"]["stopped"]
    assert [r["learning_rate"] for r in early] == [0.001, 0.001, 0.0005, 0.0005, 0.0005]
    assert not (tmp_path / "early/epoch-0006").exists()
    selected = json.loads((tmp_path / "early/selection.json").read_text())
    assert selected["epoch"] == 2 and selected["mode"] == "max"
    assert Path(selected["checkpoint"]) == (tmp_path / "early/epoch-0002").resolve()
    assert_same_checkpoint(tmp_path / "early/epoch-0005", tmp_path / "full/epoch-0005")
    summary = json.loads((tmp_path / "early/early-stop.json").read_text())
    assert summary["stopped_early"] and summary["completed_epochs"] == 5
    events = EventAccumulator(str(tmp_path / "early/tensorboard")).Reload()
    assert [x.step for x in events.Scalars("validation/early_stop_score")] == [1, 2, 3, 5]
    assert [x.value for x in events.Scalars("train/early_stop_wait")] == [0, 0, 1, 1, 2]
    resumed = run(cfg, tmp_path / "input", tmp_path / "resume", tmp_path / "early/epoch-0003")
    assert logical(resumed) == logical(early[3:])
    assert_same_checkpoint(tmp_path / "resume/epoch-0005", tmp_path / "early/epoch-0005")
    assert json.loads((tmp_path / "resume/selection.json").read_text()) == selected
    stopped = run(
        cfg, tmp_path / "input", tmp_path / "already-stopped", tmp_path / "early/epoch-0005"
    )
    assert stopped == []
    assert json.loads((tmp_path / "already-stopped/selection.json").read_text()) == selected
    altered = copy.deepcopy(cfg)
    altered["early_stop"]["patient"] = 3
    with pytest.raises(ValueError, match="identity"):
        run(altered, tmp_path / "input", tmp_path / "bad-resume", tmp_path / "early/epoch-0003")


@pytest.mark.parametrize(
    "metric,values,expected_length", [("logloss", [0.7, 0.6, 0.6, 0.65], 3), ("auc", [None] * 4, 4)]
)
def test_minimize_or_unavailable_to_max_epochs(
    tmp_path, monkeypatch, metric, values, expected_length
):
    cfg = runtime(tmp_path, 4)
    cfg["early_stop"] = options(metric, patient=1)
    override_validation(monkeypatch, values, metric)
    records = run(cfg, tmp_path / "input", tmp_path / "output")
    assert len(records) == expected_length
    selected = json.loads((tmp_path / "output/selection.json").read_text())
    if metric == "logloss":
        assert selected["epoch"] == 2 and selected["mode"] == "min"
    else:
        assert selected["status"] == "unavailable"
        assert all(r["early_stop"]["score"] is None for r in records)
        assert all(r["early_stop"]["wait"] == 0 for r in records)
        resumed = run(
            cfg, tmp_path / "input", tmp_path / "unavailable-resume", tmp_path / "output/epoch-0004"
        )
        assert resumed == []
        assert json.loads((tmp_path / "unavailable-resume/selection.json").read_text()) == selected


def test_invalid_config_fails_before_creating_output(tmp_path):
    cfg = configured(options(patient=0))
    with pytest.raises(ValueError, match="patient"):
        run(cfg, tmp_path / "nonexistent", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("available", [True, False])
def test_experiment_evaluates_selected_checkpoint_or_records_unavailable(
    tmp_path, monkeypatch, available
):
    from urllib.parse import unquote, urlsplit

    import pyarrow.parquet as pq

    from recommend_tf.train.experiment import execute
    from recommend_tf.train.models.rerank.data_utils import migrate_experiment
    from recommend_tf.train.tests.experiment_fixture import make_experiment

    source = make_experiment(tmp_path / "fixture", "deepfm", rows_per_day=4)
    source["max_epochs"] = 4
    manifest = json.loads((tmp_path / "fixture/dataset/manifest.json").read_text())
    schema = pq.read_schema(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    dtypes = {
        field.name: "str"
        if str(field.type) == "string"
        else "int64"
        if "int64" in str(field.type)
        else "float32"
        for field in schema
        if field.name != "impression_time"
    }
    cfg = migrate_experiment(
        source,
        dtypes,
        {"optimizer": "adamw", "learning_rate": 0.001, "optimizer_args": {"weight_decay": 0.0001}},
        16,
    )
    cfg["early_stop"] = options(patient=1, weight={"completion": 1})
    override_validation(monkeypatch, [0.6, 0.7, 0.7, 0.7] if available else [None] * 4)
    status = execute(cfg, tmp_path / "run")
    assert status["early_stop"]["completed_epochs"] == (3 if available else 4)
    if available:
        assert status["status"] == "completed" and status["selected_epoch"] == 2
        result = json.loads((tmp_path / "run/test/metrics.json").read_text())
        assert Path(result["checkpoint"]) == (tmp_path / "run/training/epoch-0002").resolve()
    else:
        assert status["status"] == "trained_without_selection" and status["test"] == "not_run"
        assert not (tmp_path / "run/test").exists()
