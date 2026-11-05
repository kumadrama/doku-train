import json

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm import timing
from recommend_tf.train.comm.datasvr.prepared import prepare
from recommend_tf.train.comm.training_pipeline import run
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_migration import config


def test_iterator_excludes_consumer_and_preserves_order(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: now[0])

    def values():
        for i in range(3):
            now[0] += 2
            yield i
        now[0] += 3

    with timing.session({"profiling": {"enabled": True}}, tmp_path, "test") as timer:
        seen = []
        for value in timer.iterate("input", values()):
            seen.append(value)
            now[0] += 10
    assert seen == [0, 1, 2]
    summary = json.loads(timer.summary.read_text())
    assert summary["stages"]["input"]["elapsed_seconds"] == 9
    assert summary["elapsed_seconds"] == 39


def test_failed_and_disabled_sessions(tmp_path):
    with timing.session({}, tmp_path / "off", "test") as timer:
        assert not timer.enabled
    assert not (tmp_path / "off").exists()
    with (
        pytest.raises(ValueError),
        timing.session({"profiling": {"enabled": True}}, tmp_path / "on", "test") as timer,
        timer.span("broken"),
    ):
        raise ValueError("fixture")
    assert json.loads(timer.summary.read_text())["status"] == "failed"
    assert not timing.current().enabled


def test_timing_preserves_training_and_resume(tmp_path):
    model = config()
    params = parameters(model)
    examples = [
        dict(id=i % 3, history=[0, 1], value=float(i % 3), click=i % 2, scene=0) for i in range(8)
    ]
    for split in ["train", "validation"]:
        prepare([examples], tmp_path / "input" / split, {}, params, {}, "True", "fixture", 1000000)
    cfg = dict(
        model=model,
        prepared_identity="fixture",
        dtypes=dict(id="int64", history="int64", value="float32", click="int64", scene="int64"),
        batch_size=4,
        seed=42,
        max_epochs=2,
        selection={"mode": "min", "task_losses": {"a": 1}},
        shuffle_buffer_size=4,
        intra_op_threads=1,
        inter_op_threads=1,
    )
    baseline = run(cfg, tmp_path / "input", tmp_path / "baseline")
    with timing.session({"profiling": {"enabled": True}}, tmp_path / "profile", "test") as timer:
        measured = run(cfg, tmp_path / "input", tmp_path / "measured")
        resumed = run(
            cfg, tmp_path / "input", tmp_path / "resumed", tmp_path / "baseline/epoch-0001"
        )

    def logical(records):
        return [{k: v for k, v in r.items() if k != "stage_seconds"} for r in records]

    assert logical(baseline) == logical(measured)
    assert logical(resumed) == logical(baseline[1:])
    first = tf.train.load_checkpoint(str(tmp_path / "baseline/epoch-0002/model"))
    second = tf.train.load_checkpoint(str(tmp_path / "measured/epoch-0002/model"))
    for key in first.get_variable_to_shape_map():
        np.testing.assert_array_equal(first.get_tensor(key), second.get_tensor(key))
    stages = json.loads(timer.summary.read_text())["stages"]
    assert stages["training/train_epoch/compute"]["count"] == 6
    assert stages["training/validation_epoch/input"]["count"] == 6
    assert stages["training/checkpoint"]["count"] == 3
