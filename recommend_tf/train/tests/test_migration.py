import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.common_utils import eval_expr
from recommend_tf.train.comm.datasvr.prepared import batches, organize, prepare, rows, tensor_batch
from recommend_tf.train.comm.graph_helper import make_optimizer
from recommend_tf.train.comm.nn_utils import get_loss
from recommend_tf.train.comm.training_pipeline import run
from recommend_tf.train.models.rerank.model_params import parameters

tf.disable_eager_execution()


def config():
    return dict(
        input_features={
            "item": {"features": ["id"], "from": "item"},
            "history": {
                "features": ["history"],
                "from": "history",
                "history_pooling": "attention",
                "hist_attn_query": "item",
            },
            "scalar": {"features": ["value"], "from": "item", "input_type": "scalar"},
        },
        embed_name={"id": "drama", "history": "drama"},
        embed_dim={"*": 4},
        dnn_inputs=["item", "history", "scalar"],
        fm_layers=[["item", "history"]],
        output_layer={"dims": [8], "acts": ["relu"], "layer_dropout": 0.9},
        tasks={
            name: {
                "dims": [4],
                "acts": ["relu"],
                "num_outputs": 1,
                "label": "click > 0",
                "loss_type": "logloss",
                "predict_fn": "sigmoid",
                "loss_weight": "float(scene != 47)",
                "metrics": {"auc": {"func": "auc"}, "logloss": {"func": "logloss"}},
            }
            for name in ["a", "b", "c", "d", "e", "f"]
        }
        | {
            "watch": {
                "dims": [4],
                "acts": ["relu"],
                "num_outputs": 3,
                "label": "clip(value, 0.0, 2.0)",
                "loss_type": "softmax_soft",
                "predict_fn": "expectation",
                "loss_weight": "0.3",
                "soft_label": {"func": "normal", "scale": 1, "sigma": 1},
            }
        },
    )


def test_graph_expression_and_weight_denominator():
    with tf.Graph().as_default(), tf.Session() as sess:
        features = {"item": {"scene": tf.constant([47, 1]), "click": tf.constant([1, 0])}}
        y = eval_expr("click > 0", features["item"])
        head = {
            "predict_fn": "sigmoid",
            "loss_type": "logloss",
            "loss_weight": "float(scene != 47)",
            "metrics": {"auc": {"func": "auc"}, "logloss": {"func": "logloss"}},
        }
        default, _ = get_loss(head, tf.zeros([2]), y, features)
        weighted, _ = get_loss(dict(head, loss_by_weight=True), tf.zeros([2]), y, features)
        a, b = sess.run([default, weighted])
        np.testing.assert_allclose([a, b], [np.log(2) / 2, np.log(2)], rtol=1e-6)


def test_finder_adamw_sparse_decay():
    with tf.Graph().as_default(), tf.Session() as sess:
        dense = tf.get_variable("dense", initializer=[1.0, 2.0])
        sparse = tf.get_variable("sparse", initializer=[[1.0], [2.0], [3.0]])
        opt = make_optimizer("adamw", 0.1, {"weight_decay": 0.01})
        op = opt.apply_gradients(
            [
                (tf.zeros_like(dense), dense),
                (tf.IndexedSlices(tf.zeros([1, 1]), tf.constant([1]), tf.constant([3, 1])), sparse),
            ]
        )
        sess.run(tf.global_variables_initializer())
        sess.run(op)
        a, b = sess.run([dense, sparse])
        np.testing.assert_allclose(a, [0.99, 1.98])
        np.testing.assert_allclose(b[:, 0], [1, 1.98, 3])


def test_history_rules_and_padding():
    group = {
        "h": {
            "features": ["ids"],
            "weight_feature": "weights",
            "history_pooling": "weighted_mean",
            "from": "h",
        }
    }
    opt = {
        "h": {
            "truncate": "top_weight",
            "duplicates": "sum_weights",
            "tie_break": "raw_id_ascending",
            "max_hist_size": 2,
        }
    }
    row = organize({"ids": [3, 1, 3, 2], "weights": [1.0, 2.0, 1.0, 3.0]}, {}, group, opt)
    assert row["ids"] == [2, 1] and row["weights"] == [3.0, 2.0] and row["length:h"] == 2
    params = SimpleNamespace(input_features=group)
    empty = {"ids": [], "weights": [], "length:h": 0}
    data = tensor_batch([row, empty], params, {"ids": "int64", "weights": "float32"})
    assert data["h"]["length"].tolist() == [2, 0]
    assert data["h"]["ids"].tolist() == [[2, 1], [0, 0]]


@pytest.mark.parametrize("norm", ["", "ln", "ln_l2"])
def test_prepared_integrity_shuffle_and_seven_task_train(tmp_path, norm):
    model = config()
    for layer in [model["output_layer"], *model["tasks"].values()]:
        layer["norms"] = [norm] * len(layer["dims"])
    if norm == "ln_l2":
        model.update(
            dense_use_weight_decay=True, sparse_use_weight_decay=True, sparse_weight_decay=0.00003
        )
        model["output_layer"]["regs"] = [0.0, 0.00003]
        for layer in [model["output_layer"], *model["tasks"].values()]:
            layer["norms"] = ["ln"] * len(layer["dims"])
    params = parameters(model)
    examples = [
        {
            "id": i % 3,
            "history": [0, 1][: i % 3],
            "value": float(i % 3),
            "click": i % 2,
            "scene": i % 2,
        }
        for i in range(8)
    ]
    for split in ["train", "validation", "test"]:
        prepare([examples], tmp_path / "input" / split, {}, params, {}, "True", "fixture", 1000000)
    a = list(rows(tmp_path / "input/train", "fixture"))
    assert sum(map(len, batches(iter(a), 3, 42, 4))) == 8
    cfg = dict(
        model=model,
        prepared_identity="fixture",
        dtypes={
            "id": "int64",
            "history": "int64",
            "value": "float32",
            "click": "int64",
            "scene": "int64",
        },
        batch_size=3,
        seed=42,
        max_epochs=2,
        selection={"mode": "min", "task_losses": {"a": 1.0}},
        shuffle_buffer_size=4,
        intra_op_threads=1,
        inter_op_threads=1,
    )
    metrics = run(cfg, tmp_path / "input", tmp_path / "output")
    assert len(metrics) == 2 and len(metrics[0]["train_loss"]) == 7
    if norm == "ln_l2":
        objective = metrics[0]["train_objective_batch_mean"]
        assert objective["dense_reg_loss"] > 0 and objective["sparse_reg_loss"] > 0
        np.testing.assert_allclose(
            objective["optimization_loss"],
            sum(objective[k] for k in ["data_loss", "dense_reg_loss", "sparse_reg_loss"]),
            rtol=1e-6,
        )
    resumed = run(cfg, tmp_path / "input", tmp_path / "resume", tmp_path / "output/epoch-0001")
    assert len(resumed) == 1
    for name, value in metrics[1]["validation_loss"].items():
        np.testing.assert_allclose(resumed[0]["validation_loss"][name], value, rtol=1e-6)
    shard = tmp_path / "input/train/000000.parquet"
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        list(rows(tmp_path / "input/train", "fixture"))


def test_model_intermediates_against_fixed_reference():
    from recommend_tf.train.models.rerank.modules.utils import fm_layer
    from recommend_tf.train.models.rerank.rerank_model import (
        attention_layer,
        scalar_features,
        transform_features,
    )

    with tf.Graph().as_default(), tf.Session() as sess:
        params = parameters(config())
        params.feature_transformations = [
            {
                "expr": 'bucket = cast(where(value > 1, 2, 0), "int64")',
                "apply_to": ["item"],
                "apply_to_features": [],
            }
        ]
        source = {"item": {"value": tf.constant([0.0, 2.0])}}
        changed = transform_features(source, params)
        scalars = scalar_features(changed, tf.estimator.ModeKeys.TRAIN, params)
        inputs = {
            "item": tf.constant([[1.0, 0.0], [1.0, 0.0]]),
            "history": tf.constant([[[1.0, 0.0], [0.0, 1.0]], [[9.0, 9.0], [9.0, 9.0]]]),
        }
        attention_layer(
            params, tf.estimator.ModeKeys.TRAIN, inputs, {"history": tf.constant([2, 0])}, {}, {}
        )
        pair = fm_layer(params, [tf.constant([[1.0, 2.0]]), tf.constant([[3.0, 4.0]])])
        bucket, scalar, attn, fm = sess.run(
            [changed["item"]["bucket"], scalars["scalar"], inputs["history"], pair]
        )
        assert "bucket" not in source["item"]
        np.testing.assert_array_equal(bucket, [0, 2])
        np.testing.assert_array_equal(scalar, [[0], [2]])
        np.testing.assert_allclose(attn, [[np.e / (np.e + 1), 1 / (np.e + 1)], [0, 0]], rtol=1e-6)
        np.testing.assert_array_equal(fm, [[3, 8]])


def test_graph_weighted_mean_empty_and_gradients():
    from recommend_tf.train.models.rerank.rerank_model import attention_layer

    with tf.Graph().as_default(), tf.Session() as sess:
        params = SimpleNamespace(
            input_features={
                "h": {"history_pooling": "weighted_mean", "from": "h", "weight_feature": "w"}
            }
        )
        value = tf.Variable([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        inputs = {"h": value}
        attention_layer(
            params,
            "train",
            inputs,
            {"h": tf.constant([2, 0])},
            {},
            {"h": {"w": tf.constant([[1.0, 3.0], [0.0, 0.0]])}},
        )
        grad = tf.gradients(tf.reduce_sum(inputs["h"]), value)[0]
        sess.run(tf.global_variables_initializer())
        out, g = sess.run([inputs["h"], grad])
        np.testing.assert_allclose(out, [[2.5, 3.5], [0, 0]])
        np.testing.assert_allclose(g, [[[0.25, 0.25], [0.75, 0.75]], [[0, 0], [0, 0]]])


def test_reject_torch_decay_and_unfinished_preparation(tmp_path):
    with pytest.raises(ValueError, match="implicitly"):
        make_optimizer("adam", 0.01, {"weight_decay": 0.001})
    with pytest.raises(ValueError, match="explicit"):
        make_optimizer("adamw", 0.01, {})
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        list(rows(incomplete, "x"))
    with pytest.raises(ValueError, match="Unsupported"):
        parameters(dict(config(), use_hstu_module=True))


def test_metrics_auc_ties_and_single_class(tmp_path):
    from recommend_tf.train.comm.eval.metrics import Metrics

    task = {"metrics": {"auc": {"func": "auc"}, "mse": {"func": "mse"}}}
    metric = Metrics(tmp_path / "metrics.sqlite", {"a": task, "b": task})
    metric.add("a", [0.5, 0.5, 1.0], {"auc": [0, 1, 1], "mse": [0, 1, 1]})
    metric.add("b", [0.1, 0.2], {"auc": [0, 0], "mse": [0, 0]})
    values = metric.finish()
    metric.close()
    assert values["a"]["auc"]["value"] == 0.75
    assert values["b"]["auc"]["value"] is None
    assert values["b"]["auc"]["reason"] == "single_class"


def test_manifest_entry_with_explicit_finder_optimizer(tmp_path):
    import pyarrow.parquet as pq

    from recommend_tf.train.models.rerank.data_utils import migrate_experiment
    from recommend_tf.train.tests.experiment_fixture import make_experiment
    from recommend_tf.train.train_rerank import train

    source = make_experiment(tmp_path / "fixture", "deepfm", rows_per_day=5)
    source["max_epochs"] = 1
    manifest = json.loads((tmp_path / "fixture/dataset/manifest.json").read_text())
    from urllib.parse import unquote, urlsplit

    schema = pq.read_schema(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    dtypes = {}
    for field in schema:
        name = field.name
        if name == "impression_time":
            continue
        dtype = str(field.type)
        dtypes[name] = "str" if dtype == "string" else "int64" if "int64" in dtype else "float32"
    cfg = migrate_experiment(source, dtypes, {"optimizer": "adam", "learning_rate": 0.001}, 16)
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    assert (tmp_path / "run/input/record.json").exists()
    shard = next((tmp_path / "run/prepared/train").glob("*.parquet"))
    assert "impression_time" not in pq.read_schema(shard).names
    assert (tmp_path / "run/training/lineage.json").exists()
    from recommend_tf.train.comm.training_pipeline import evaluate_checkpoint

    runtime = json.loads((tmp_path / "run/resolved-config.json").read_text())
    measured = evaluate_checkpoint(
        runtime,
        tmp_path / "run/prepared",
        tmp_path / "run/training/epoch-0001",
        tmp_path / "evaluation",
    )
    assert measured["split"] == "test" and len(measured["loss"]) == len(cfg["model"]["tasks"])
    # Adapter never silently translates the old Torch weight_decay.
    assert "weight_decay" not in cfg["model"].get("optimizer_args", {})


def test_padding_does_not_touch_sparse_rows():
    from recommend_tf.train.comm.graph_helper import SPARSE_VARIABLES
    from recommend_tf.train.comm.vocab_manager import lookup

    with tf.Graph().as_default(), tf.Session() as sess:
        params = parameters(config(), {"drama": {"dtype": "int64", "keys": [0, 1]}})
        value = lookup(
            tf.constant([[0, 1], [0, 0]], tf.int64),
            "drama",
            params,
            tf.constant([[True, True], [False, False]]),
        )
        variable = tf.get_collection(SPARSE_VARIABLES)[0]
        gradient = tf.gradients(tf.reduce_sum(value), variable)[0]
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        out, indices = sess.run([value, gradient.indices])
        assert indices.tolist() == [1, 2]
        np.testing.assert_array_equal(out[1], np.zeros((2, 4)))


def test_predict_graph_needs_no_labels_and_oov_is_shared():
    from recommend_tf.train.comm.graph_helper import SPARSE_VARIABLES
    from recommend_tf.train.models.rerank.rerank_model import model_fn

    cfg = config()
    cfg["broadcast_feat"] = "item/id"
    state = {"drama": {"dtype": "int64", "keys": [0, 1]}}
    with tf.Graph().as_default(), tf.Session() as sess:
        params = parameters(cfg, state)
        features = {
            "item": {"id": tf.constant([99, 0], tf.int64), "value": tf.constant([0.5, 1.0])},
            "history": {
                "history": tf.constant([[99], [0]], tf.int64),
                "length": tf.constant([1, 0], tf.int64),
            },
        }
        result = model_fn(features, None, tf.estimator.ModeKeys.PREDICT, params)
        assert len(tf.get_collection(SPARSE_VARIABLES)) == 1
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        predictions = sess.run(result["predictions"])
        assert len(predictions) == 7 and all(np.isfinite(x).all() for x in predictions.values())
        assert "loss" not in result


def test_prepared_projection_skips_audit_timestamp(tmp_path):
    from datetime import datetime, timezone

    from recommend_tf.train.comm.datasvr.prepared import input_columns

    params = parameters(config())
    examples = [
        {
            "id": 1,
            "history": [1],
            "value": 1.0,
            "click": 1,
            "scene": 1,
            "impression_time": datetime(2026, 10, 1, tzinfo=timezone.utc),
        }
    ]
    prepare([examples], tmp_path / "input", {}, params, {}, "True", "fixture", 1000000)
    dtypes = {
        "id": "int64",
        "history": "int64",
        "value": "float32",
        "click": "int64",
        "scene": "int64",
    }
    full = list(rows(tmp_path / "input", "fixture"))
    projected = list(rows(tmp_path / "input", "fixture", columns=input_columns(params, dtypes)))
    assert "impression_time" not in projected[0]
    assert projected[0]["length:history"] == 1
    a, b = tensor_batch(full, params, dtypes), tensor_batch(projected, params, dtypes)
    for group in a:
        for field in a[group]:
            np.testing.assert_array_equal(a[group][field], b[group][field])
    assert "impression_time" in next(
        rows(tmp_path / "input", "fixture", columns=["impression_time"])
    )


def test_columnar_rows_preserve_values_and_shuffle(tmp_path):
    from datetime import datetime, timezone

    from recommend_tf.train.comm.datasvr.prepared import ColumnRow

    params = parameters(config())
    examples = [
        {
            "id": i,
            "history": [i, i + 1] if i % 3 else [],
            "value": float(i),
            "click": i % 2,
            "scene": 1,
            "text": "long_value" if i % 2 else "x",
            "optional": None if i % 2 else i,
            "impression_time": datetime(2026, 10, 1, tzinfo=timezone.utc),
        }
        for i in range(1100)
    ]
    prepare([examples], tmp_path / "input", {}, params, {}, "True", "fixture", 10000000)
    actual = list(rows(tmp_path / "input", "fixture"))
    assert isinstance(actual[0], ColumnRow)
    for i, row in enumerate(actual):
        for key in ("id", "value", "click", "scene", "text", "optional", "impression_time"):
            assert row[key] == examples[i][key]
        np.testing.assert_array_equal(row["history"], examples[i]["history"])
    # Same random draws, insertion/removal order, batch boundaries, and final drain.
    import random

    def reference(source, size, seed, capacity):
        rng = random.Random(seed)
        buffer, batch, result = [], [], []
        for row in source:
            buffer.append(row)
            if len(buffer) < capacity:
                continue
            index = rng.randrange(len(buffer)) if seed is not None else 0
            batch.append(buffer.pop(index)["id"])
            if len(batch) == size:
                result.append(batch)
                batch = []
        while buffer:
            index = rng.randrange(len(buffer)) if seed is not None else 0
            batch.append(buffer.pop(index)["id"])
            if len(batch) == size:
                result.append(batch)
                batch = []
        if batch:
            result.append(batch)
        return result

    for seed in [None, 42, 43]:
        for capacity in [1, 32, 8192]:
            expected = reference(examples, 257, seed, capacity)
            measured = [
                [row["id"] for row in batch] for batch in batches(iter(actual), 257, seed, capacity)
            ]
            assert measured == expected


def test_memory_checks_are_bounded_and_still_enforced(monkeypatch):
    from recommend_tf.train.comm.datasvr import prepared

    calls = []
    monkeypatch.setattr(prepared, "check_memory", lambda: calls.append(True))
    assert sum(len(x) for x in batches(iter(range(5000)), 2048, 42, 8192)) == 5000
    assert len(calls) == 8  # 5 input intervals + 3 output batches.

    def fail():
        raise MemoryError("budget exceeded")

    monkeypatch.setattr(prepared, "check_memory", fail)
    with pytest.raises(MemoryError, match="budget exceeded"):
        next(batches(iter(range(5000)), 2048, 42, 8192))


def test_column_preparation_matches_row_rules_and_fitting_batches(tmp_path):
    from recommend_tf.train.comm.datasvr.prepared import (
        column_batches,
        organize_columns,
        tensor_columns,
    )

    params = parameters(config())
    defaults = {"id": 0, "value": 0.0, "history": []}
    policies = {"history": {"max_hist_size": 2, "truncate": "tail", "duplicates": "preserve"}}
    source = [
        {"id": 2, "value": None, "history": [], "click": 0, "scene": 1},
        {"id": 3, "value": 1.0, "history": [3, 2, 1], "click": 1, "scene": 1},
        {"id": None, "value": 2.0, "history": None, "click": 0, "scene": 47},
        {"id": 4, "value": 3.0, "history": [4], "click": 1, "scene": 1},
    ]
    blocks = [source[:1], source[1:3], source[3:]]
    columns = [{key: [row[key] for row in block] for key in source[0]} for block in blocks]
    for block, cols in zip(blocks, columns):
        organized = organize_columns(cols, defaults, params.input_features, policies)
        expected = [organize(row, defaults, params.input_features, policies) for row in block]
        assert [dict(zip(organized, values)) for values in zip(*organized.values())] == expected
    for name, data in [("row", blocks), ("column", columns)]:
        prepare(data, tmp_path / name, defaults, params, policies, "scene != 47", "x", 10**8)
    left, right = list(rows(tmp_path / "row", "x")), list(rows(tmp_path / "column", "x"))
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
    dtypes = dict(id="int64", value="float32", history="int64", click="int64", scene="int64")
    names = list(dtypes) + ["length:history"]
    old = list(batches(rows(tmp_path / "column", "x", names), 2))
    new = list(column_batches(tmp_path / "column", "x", names, 2))
    assert len(old) == len(new) == 2
    for before, after in zip(old, new):
        a, b = tensor_batch(before, params, dtypes), tensor_columns(after, params, dtypes)
        for group in a:
            for field in a[group]:
                assert a[group][field].dtype == b[group][field].dtype
                np.testing.assert_array_equal(a[group][field], b[group][field])
    bad = dict(columns[0], history=[[None]], scene=[47])
    with pytest.raises(ValueError, match="null elements"):
        prepare([bad], tmp_path / "bad", defaults, params, policies, "scene != 47", "x", 10**8)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_reader_vector_time_validation(tmp_path, unit):
    from datetime import datetime, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq

    from recommend_tf.train.comm import timing
    from recommend_tf.train.comm.datasvr.published_dataset import PublishedDataset

    reader = PublishedDataset.__new__(PublishedDataset)
    reader.config = {
        "features": ["id"],
        "feature_default_values": {},
        "batch_size": 8,
        "execution": {"arrow_threads": 1},
    }
    reader.daily_fields = {}
    reader.timer = timing.current()
    partition = {
        "uri": "fixture",
        "event_time_from": "2026-10-01T00:00:00Z",
        "event_time_to": "2026-10-02T00:00:00Z",
    }
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 2, tzinfo=timezone.utc)
    path = tmp_path / "input.parquet"
    for value, valid in [(start, True), (end, False), (None, False)]:
        pq.write_table(
            pa.table(
                {"id": [1], "impression_time": pa.array([value], type=pa.timestamp(unit, tz="UTC"))}
            ),
            path,
        )
        if valid:
            assert list(reader._columns_in_shard(partition, path, {"id"})) == [{"id": [1]}]
            explicit = list(reader._columns_in_shard(partition, path, {"id", "impression_time"}))
            assert explicit[0]["impression_time"] == [start]
        else:
            with pytest.raises(ValueError, match="outside declared"):
                list(reader._columns_in_shard(partition, path, {"id"}))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_finder_ln_forward_gradients_and_eval_reuse(shape):
    from recommend_tf.train.models.rerank.rerank_model import _get_norm

    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7
    # Include a constant row to exercise epsilon and zero variance.
    data.reshape(-1, shape[-1])[0] = 2.0
    weights = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 13
    with tf.Graph().as_default(), tf.Session() as session:
        x = tf.placeholder(tf.float32, shape)
        fn = _get_norm(x, "fixture", "ln", tf.estimator.ModeKeys.TRAIN)
        actual = fn(x)
        eval_fn = _get_norm(x, "fixture", "ln", tf.estimator.ModeKeys.EVAL)
        variables = tf.trainable_variables()
        assert len(variables) == 2
        gamma = next(v for v in variables if "gamma" in v.name)
        beta = next(v for v in variables if "beta" in v.name)
        # Independent expression with explicit centering and division.
        centered = x - tf.reduce_mean(x, axis=-1, keepdims=True)
        reference = centered / tf.sqrt(tf.reduce_mean(centered**2, axis=-1, keepdims=True) + 1e-6)
        reference = reference * gamma + beta
        gradients = tf.gradients(tf.reduce_sum(actual * weights), [x, gamma, beta])
        expected_gradients = tf.gradients(tf.reduce_sum(reference * weights), [x, gamma, beta])
        session.run(tf.global_variables_initializer())
        session.run([gamma.assign([0.8, 1.2, 1.1, 0.9]), beta.assign([0.1, 0.2, -0.1, -0.2])])
        a, b, e, ga, gb = session.run(
            [actual, reference, eval_fn(x), gradients, expected_gradients], {x: data}
        )
        # Finder uses x*inv + (beta-mean*inv): constant nonzero rows
        # incur float32 cancellation at inv≈1000, unlike centered division.
        af, bf = a.reshape(-1, shape[-1]), b.reshape(-1, shape[-1])
        np.testing.assert_allclose(af[0], bf[0], rtol=0, atol=2e-4)
        np.testing.assert_allclose(af[1:], bf[1:], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(a, e)
        for left, right in zip(ga, gb):
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4)
    for norms in [["bn"], ["ln", "ln"]]:
        invalid = config()
        invalid["output_layer"]["norms"] = norms
        with pytest.raises(ValueError, match="supported norm"):
            parameters(invalid)
