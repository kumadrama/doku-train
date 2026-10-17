import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recommend.train.comm.auc_events import emit_events, record_events


def inputs():
    config = {
        "model_name": "test_model",
        "output": {"run_id": "run1"},
        "tasks": {
            "click": {"metrics": {"rank": {"func": "auc"}, "loss": {"func": "logloss"}}},
            "watch": {"metrics": {"rank": {"func": "regression_auc"}}},
        },
    }
    metrics = {
        "click": {"metrics": {"rank": {"value": 0.7}, "loss": {"value": 0.4}}},
        "watch": {"metrics": {"rank": {"value": None}}},
    }
    return config, metrics


def test_freezes_only_auc_with_null_and_shanghai_day(tmp_path):
    config, metrics = inputs()
    now = datetime(2026, 10, 17, 16, 5, tzinfo=UTC)
    events = record_events(tmp_path, config, metrics, completed_at=now)
    assert len(events) == 2
    assert events[0]["metric"] == "rank"
    assert events[1]["kind"] == "regression_auc" and events[1]["auc"] is None
    assert events[0]["day_start_ms"] == int(
        datetime(2026, 10, 17, 16, tzinfo=UTC).timestamp() * 1000
    )
    again = record_events(tmp_path, config, metrics)
    assert again == events
    stream = io.StringIO()
    emit_events(events, stream=stream)
    assert [json.loads(line) for line in stream.getvalue().splitlines()] == events
    metrics["click"]["metrics"]["rank"]["value"] = 0.8
    with pytest.raises(ValueError, match="changed"):
        record_events(tmp_path, config, metrics)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_auc_does_not_create_event(tmp_path, value):
    config, metrics = inputs()
    metrics["click"]["metrics"]["rank"]["value"] = value
    with pytest.raises(ValueError):
        record_events(tmp_path, config, metrics)
    assert not (tmp_path / "test-auc-events.json").exists()


def test_missing_declared_metric_is_not_a_null_auc(tmp_path):
    config, metrics = inputs()
    del metrics["click"]["metrics"]["rank"]
    with pytest.raises(KeyError):
        record_events(tmp_path, config, metrics)


def test_versioned_wire_fixture(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures/test-auc-v1.json"
    for index, event in enumerate(json.loads(fixture.read_text())):
        root = tmp_path / str(index)
        root.mkdir()
        config = {
            "model_name": event["model_name"],
            "output": {"run_id": event["run_id"]},
            "tasks": {event["task"]: {"metrics": {event["metric"]: {"func": event["kind"]}}}},
        }
        metrics = {event["task"]: {"metrics": {event["metric"]: {"value": event["auc"]}}}}
        completed = datetime.fromtimestamp(event["completed_at_ms"] / 1000, UTC)
        assert record_events(root, config, metrics, completed_at=completed) == [event]


@pytest.mark.parametrize("failure", ["train", "test", "observer"])
def test_reporting_does_not_mask_execution_or_change_publication(
    tmp_path, monkeypatch, capsys, failure
):
    from recommend.train import eval_rerank, run, train_rerank
    from recommend.train.tests.smoke.test_tree_publication import daily_fixture

    config, daily = daily_fixture(tmp_path)
    for rule in config["publication"].values():
        rule["minimum"] = 1.0
    root = tmp_path / "fake-run"
    root.mkdir()
    observed = []

    def train(value):
        if failure == "train":
            raise RuntimeError("training failure")
        return {"run_dir": str(root)}

    def evaluate(path):
        if failure == "test":
            raise RuntimeError("evaluation failure")
        return {
            task: {"metrics": {key: {"value": 0.7} for key in options["metrics"]}}
            for task, options in config["tasks"].items()
        }

    def observer(events):
        observed.extend(events)
        raise OSError("private transport details must not be logged")

    monkeypatch.setattr(train_rerank, "execute_train", train)
    monkeypatch.setattr(eval_rerank, "execute_test", evaluate)
    monkeypatch.setattr(run, "archive_unpublished_run", lambda *args: None)
    kwargs = {
        "daily_root": daily.root,
        "frozen_root": (tmp_path / "frozen").as_uri(),
        "cutoff": "2026-10-08",
        "artifact_root": (tmp_path / "artifacts").as_uri(),
        "export_root": tmp_path / "exports",
        "test_observer": observer,
    }
    if failure != "observer":
        with pytest.raises(RuntimeError):
            run.daily_run(config, **kwargs)
        assert not observed and not (root / "test-auc-events.json").exists()
    else:
        result = run.daily_run(config, **kwargs)
        assert result["status"] == "rejected" and observed
        assert (
            json.loads((Path(result["run_dir"]) / "test-auc-events.json").read_text()) == observed
        )
        error = json.loads(capsys.readouterr().err)
        assert error == {"event": "test_auc_reporting_failed", "error_type": "OSError"}
