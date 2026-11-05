import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.datasvr.prepared import ColumnRow, batches, prepare, rows, tensor_batch
from recommend_tf.train.comm.input_pipeline import InputPipeline, options, prefetch
from recommend_tf.train.comm.training_pipeline import run
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_migration import config


def equal(left, right):
    assert left.keys() == right.keys()
    for group in left:
        assert left[group].keys() == right[group].keys()
        for field in left[group]:
            np.testing.assert_array_equal(left[group][field], right[group][field])
            assert left[group][field].dtype == right[group][field].dtype


def fixture():
    params = SimpleNamespace(
        input_features={
            "item": {"features": ["id"], "from": "item"},
            "history": {
                "features": ["history"],
                "from": "history",
                "history_pooling": "mean",
                "weight_feature": "weights",
            },
        }
    )
    dtypes = dict(id="int64", name="str", history="int64", weights="float32")
    blocks = [
        {
            "id": np.array([2**60 + 1, 2, 3]),
            "name": np.array(["long name", "中", ""]),
            "history": [[1, 2], [], [3]],
            "weights": [[0.1, 0.2], [], [0.3]],
            "length:history": np.array([2, 0, 1]),
        },
        {
            "id": np.array([4, 5]),
            "name": np.array(["x", "abc"]),
            "history": [[], [4, 5, 6]],
            "weights": [[], [0.4, 0.5, 0.6]],
            "length:history": np.array([0, 3]),
        },
    ]
    batch = [
        ColumnRow(blocks[b], i, ("fixture", b)) for b, i in [(1, 0), (0, 2), (1, 1), (0, 0), (0, 1)]
    ]
    return params, dtypes, batch


@pytest.mark.parametrize("limit", [0, 1, 10000])
def test_columns_cache_exact_ragged_string_large_int_and_order(limit):
    params, dtypes, batch = fixture()
    pipe = InputPipeline(
        {"input_pipeline": {"batch_encoding": "columns", "encoded_cache_bytes": limit}},
        params,
        dtypes,
    )
    for chosen in [batch, batch[::-1], [batch[0]], batch[:2], batch]:
        equal(pipe.encode(chosen), tensor_batch(chosen, params, dtypes))
        assert pipe.cache_bytes <= limit
    if limit == 10000:
        assert pipe.hits > 0
    else:
        assert pipe.hits == 0


def test_dtype_promotion_does_not_round_identifiers():
    params = SimpleNamespace(input_features={})
    batch = [ColumnRow({"id": np.array([2**60 + 1])}, 0), ColumnRow({"id": np.array([2.0])}, 0)]
    pipe = InputPipeline({"input_pipeline": {"batch_encoding": "columns"}}, params, {"id": "int64"})
    equal(pipe.encode(batch), tensor_batch(batch, params, {"id": "int64"}))


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": 1},
        {"prefetch_batches": -1},
        {"prefetch_batches": True},
        {"encoded_cache_bytes": 5},
        {"batch_encoding": "wrong"},
    ],
)
def test_invalid_configuration(value):
    with pytest.raises(ValueError):
        options({"input_pipeline": value})


def test_prefetch_fifo_failure_and_cancel():
    with prefetch(iter(range(20)), 2) as stream:
        assert list(stream) == list(range(20))

    def broken():
        yield 1
        raise ValueError("producer failed")

    with prefetch(broken(), 1) as stream:
        assert next(stream) == 1
        with pytest.raises(ValueError, match="producer failed"):
            next(stream)
    with prefetch(iter(range(1000000)), 1) as stream:
        assert next(stream) == 0
    assert not any(t.name == "tf-input-prefetch" for t in threading.enumerate())


@pytest.mark.parametrize(
    "settings,intra_threads",
    [
        ({"batch_encoding": "columns"}, 1),
        ({"batch_encoding": "columns", "encoded_cache_bytes": 1000000}, 1),
        ({"prefetch_batches": 2}, 1),
        ({"batch_encoding": "columns", "encoded_cache_bytes": 1000000, "prefetch_batches": 2}, 1),
        pytest.param(
            {"batch_encoding": "columns", "prefetch_batches": 2},
            1,
            id="columns_prefetch_one_thread",
        ),
        pytest.param(
            {"batch_encoding": "columns", "prefetch_batches": 2},
            4,
            id="columns_prefetch_four_threads",
        ),
    ],
)
def test_training_checkpoint_and_resume_consistency(tmp_path, settings, intra_threads):
    model = config()
    params = parameters(model)
    examples = [
        dict(id=i % 3, history=[0, 1][: i % 3], value=float(i % 3), click=i % 2, scene=0)
        for i in range(11)
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
        shuffle_buffer_size=5,
        intra_op_threads=intra_threads,
        inter_op_threads=1,
    )
    base = run(cfg, tmp_path / "input", tmp_path / "base")
    cfg["input_pipeline"] = settings
    candidate = run(cfg, tmp_path / "input", tmp_path / "candidate")
    resumed = run(cfg, tmp_path / "input", tmp_path / "resume", tmp_path / "candidate/epoch-0001")

    def logical(records):
        return [{k: v for k, v in r.items() if k != "stage_seconds"} for r in records]

    assert logical(base) == logical(candidate)
    assert logical(candidate[1:]) == logical(resumed)
    baseline = tf.train.load_checkpoint(str(tmp_path / "base/epoch-0002/model"))
    for folder in ["candidate", "resume"]:
        checkpoint = tf.train.load_checkpoint(str(tmp_path / folder / "epoch-0002/model"))
        for key in baseline.get_variable_to_shape_map():
            np.testing.assert_array_equal(baseline.get_tensor(key), checkpoint.get_tensor(key))
    if settings.get("encoded_cache_bytes"):
        assert (
            json.loads((tmp_path / "candidate/input-pipeline.json").read_text())["cache_hits"] > 0
        )

    # No sample or batch boundary changes when prefetch runs ahead of the consumer.
    def source():
        return batches(rows(tmp_path / "input/train", "fixture"), 4, 42, 5)

    pipe = InputPipeline(cfg, params, cfg["dtypes"])
    expected = [tensor_batch(b, params, cfg["dtypes"]) for b in source()]
    with pipe.batches(source()) as stream:
        actual = list(stream)
    assert len(expected) == len(actual)
    for a, b in zip(actual, expected):
        equal(a, b)
