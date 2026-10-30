import copy
import json

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm import graph_helper
from recommend_tf.train.comm.datasvr.prepared import prepare
from recommend_tf.train.comm.learning_rate import learning_rate_for_epoch, validate_schedules
from recommend_tf.train.comm.training_pipeline import run
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_migration import config


def schedule(points, kind="piecewise_constant"):
    return {"type": kind, "points": [{"epoch": e, "learning_rate": lr} for e, lr in points]}


@pytest.mark.parametrize(
    "points,expected",
    [
        ([(1, 0.001), (11, 0.0005), (21, 0.0002)], [0.001] * 10 + [0.0005] * 10 + [0.0002] * 20),
        ([(1, 0.001), (6, 0.0005), (16, 0.0002)], [0.001] * 5 + [0.0005] * 10 + [0.0002] * 25),
    ],
)
def test_requested_schedules(points, expected):
    model = {"learning_rate": 0.001, "learning_rate_schedule": schedule(points)}
    validate_schedules(model, 40)
    assert [learning_rate_for_epoch(model, e) for e in range(1, 41)] == expected


def test_linear_and_fixed():
    model = {
        "learning_rate": 0.001,
        "learning_rate_schedule": schedule([(1, 0.001), (3, 0.0002)], "piecewise_linear"),
    }
    validate_schedules(model, 4)
    assert [learning_rate_for_epoch(model, e) for e in range(1, 5)] == pytest.approx(
        [0.001, 0.0006, 0.0002, 0.0002]
    )
    assert learning_rate_for_epoch({}, 20) == 0.001
    with pytest.raises(ValueError):
        learning_rate_for_epoch({}, 0)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"type": "cosine", "points": []},
        schedule([]),
        schedule([(2, 0.001)]),
        schedule([(1, 0.002)]),
        schedule([(1, 0.001), (1, 0.0005)]),
        schedule([(1, 0.001), (41, 0.0005)]),
        schedule([(True, 0.001)]),
        schedule([(1, 0.001), (2, float("nan"))]),
        schedule([(1, 0.001), (2, float("inf"))]),
        schedule([(1, 0.001), (2, 0)]),
        schedule([(1, 0.001), (2, -1)]),
        schedule([(1, 0.001), (2, True)]),
        {"type": "piecewise_constant", "points": [{"epoch": 1, "learning_rate": 0.001, "typo": 1}]},
    ],
)
def test_invalid_schedules_fail_closed(bad):
    with pytest.raises(ValueError):
        validate_schedules({"learning_rate": 0.001, "learning_rate_schedule": bad}, 40)


def test_sparse_schedule_requires_optimizer():
    with pytest.raises(ValueError, match="sparse_optimizer"):
        validate_schedules({"sparse_learning_rate_schedule": schedule([(1, 0.001)])})


def tensors(path):
    reader = tf.train.load_checkpoint(str(path / "model"))
    return {k: reader.get_tensor(k) for k in reader.get_variable_to_shape_map()}


def assert_same_checkpoint(a, b):
    aa, bb = tensors(a), tensors(b)
    assert aa.keys() == bb.keys()
    for k in aa:
        np.testing.assert_array_equal(aa[k], bb[k], err_msg=k)


def logical(rows):
    return [{k: v for k, v in x.items() if k != "stage_seconds"} for x in rows]


@pytest.mark.parametrize("separate_sparse", [False, True])
def test_runtime_fixed_parity_schedule_and_resume(tmp_path, monkeypatch, separate_sparse):
    model = config()
    model.update(optimizer="adamw", learning_rate=0.001, optimizer_args={"weight_decay": 0.0001})
    if separate_sparse:
        model.update(
            sparse_optimizer="adamw",
            sparse_learning_rate=0.001,
            sparse_optimizer_args={"weight_decay": 0.0001},
        )
    examples = [
        dict(id=i % 3, history=[0, 1][: i % 3], value=float(i % 3), click=i % 2, scene=i % 2)
        for i in range(9)
    ]
    for split in ("train", "validation"):
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
    cfg = dict(
        model=model,
        prepared_identity="fixture",
        dtypes=dict(id="int64", history="int64", value="float32", click="int64", scene="int64"),
        batch_size=3,
        seed=42,
        max_epochs=3,
        selection={"mode": "min", "task_losses": {"a": 1.0}},
        shuffle_buffer_size=4,
        intra_op_threads=1,
        inter_op_threads=1,
    )
    fixed = run(cfg, tmp_path / "input", tmp_path / "fixed")
    constant = copy.deepcopy(cfg)
    constant["model"]["learning_rate_schedule"] = schedule([(1, 0.001)])
    same = run(constant, tmp_path / "input", tmp_path / "constant")
    assert logical(fixed) == logical(same)
    assert_same_checkpoint(tmp_path / "fixed/epoch-0003", tmp_path / "constant/epoch-0003")
    changed = copy.deepcopy(cfg)
    changed["model"]["learning_rate_schedule"] = schedule([(1, 0.001), (2, 0.0005), (3, 0.0002)])
    if separate_sparse:
        changed["model"]["sparse_learning_rate_schedule"] = schedule([(1, 0.001), (2, 0.0003)])
    seen = []
    original = graph_helper.make_optimizer

    def capture(name, rate, args):
        seen.append(rate)
        return original(name, rate, args)

    monkeypatch.setattr(graph_helper, "make_optimizer", capture)
    history = run(changed, tmp_path / "input", tmp_path / "scheduled")
    assert seen == (
        [0.001, 0.001, 0.0005, 0.0003, 0.0002, 0.0003]
        if separate_sparse
        else [0.001, 0.0005, 0.0002]
    )
    assert [x["learning_rate"] for x in history] == [0.001, 0.0005, 0.0002]
    assert logical(history[:1]) == logical(fixed[:1])
    assert history[-1]["train_loss"] != fixed[-1]["train_loss"]
    for epoch in (1, 2):
        resumed = run(
            changed,
            tmp_path / "input",
            tmp_path / f"resume{epoch}",
            tmp_path / f"scheduled/epoch-{epoch:04d}",
        )
        assert logical(resumed) == logical(history[epoch:])
        assert_same_checkpoint(
            tmp_path / f"resume{epoch}/epoch-0003", tmp_path / "scheduled/epoch-0003"
        )
    altered = copy.deepcopy(changed)
    altered["model"]["learning_rate_schedule"]["points"][1]["learning_rate"] = 0.0004
    with pytest.raises(ValueError, match="identity"):
        run(altered, tmp_path / "input", tmp_path / "bad-resume", tmp_path / "scheduled/epoch-0001")
    saved = json.loads((tmp_path / "scheduled/epoch-metrics/0003.json").read_text())
    assert saved["learning_rate"] == 0.0002
    rates = {}
    for event_file in (tmp_path / "scheduled/tensorboard").glob("events.*"):
        for event in tf.train.summary_iterator(str(event_file)):
            for value in event.summary.value:
                if value.tag == "train/learning_rate":
                    rates[event.step] = value.simple_value
    assert rates == pytest.approx({1: 0.001, 2: 0.0005, 3: 0.0002})
