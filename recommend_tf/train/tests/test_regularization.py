import random
from copy import deepcopy

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.graph_helper import optimize_loss
from recommend_tf.train.comm.regularization import loss_terms
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.models.rerank.rerank_model import embedded_features, model_fn
from recommend_tf.train.tests.test_migration import config


def regularized():
    cfg = config()
    cfg.update(
        dense_use_weight_decay=True,
        sparse_use_weight_decay=True,
        sparse_weight_decay=0.2,
        sparse_weight_decay_vars=["drama"],
    )
    cfg["output_layer"]["regs"] = [0.0, 0.1]
    return cfg


def inputs():
    return {
        "item": {
            "id": tf.constant([10, 10], tf.int64),
            "value": tf.constant([1.0, 2.0]),
            "click": tf.constant([0, 1]),
            "scene": tf.constant([1, 1]),
        },
        "history": {
            "history": tf.constant([[10, 20, 999], [999, 999, 999]], tf.int64),
            "length": tf.constant([2, 0]),
        },
    }


def vocab():
    return {"drama": {"dtype": "int64", "keys": [10, 20]}}


def test_occurrences_shared_table_padding_and_sparse_gradient():
    with tf.Graph().as_default(), tf.Session() as sess:
        params = parameters(regularized(), vocab())
        _, _, _, embeddings = embedded_features(inputs(), "train", params)
        term = loss_terms(params, embeddings, [tf.constant(0.0)])["sparse_reg_loss"]
        w = tf.trainable_variables()[0]
        grad = tf.gradients(term, w)[0]
        assert isinstance(grad, tf.IndexedSlices)
        dense_grad = tf.convert_to_tensor(grad)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(w.assign([[9.0] * 4, [1.0] * 4, [2.0] * 4]))
        loss, gradient = sess.run([term, dense_grad])
        # A twice as candidate and once in history; B once; padding OOV absent.
        np.testing.assert_allclose(loss, 0.2 / 2 * (3 * 4 + 16))
        np.testing.assert_allclose(gradient, [[0.0] * 4, [0.6] * 4, [0.4] * 4])
        train = optimize_loss(term, params)
        # Adam slots were created after initialization.
        sess.run(tf.variables_initializer([v for v in tf.global_variables() if v is not w]))
        sess.run(train)
        updated = sess.run(w)
        np.testing.assert_array_equal(updated[0], [9.0] * 4)
        assert np.all(updated[1:] < [[1.0] * 4, [2.0] * 4])


def test_dense_scope_objective_and_eval_reuse():
    with tf.Graph().as_default(), tf.Session() as sess:
        p = parameters(regularized(), vocab())
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            result = model_fn(inputs(), None, "train", p)
        kernels = [
            v
            for v in tf.trainable_variables()
            if v.name.endswith("kernel:0") and "output" in v.name
        ]
        # Collection identifies the configured hidden kernel, not logits or bias.
        terms = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        assert len(terms) == 1
        gradients = tf.gradients(
            result["objective_losses"]["dense_reg_loss"], tf.trainable_variables()
        )
        active = [(v, g) for v, g in zip(tf.trainable_variables(), gradients) if g is not None]
        assert len(active) == 1 and active[0][0].name.endswith("kernel:0")
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            evaluation = model_fn(inputs(), None, "eval", p)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        values, w, g = sess.run([result["objective_losses"], active[0][0], active[0][1]])
        np.testing.assert_allclose(values["dense_reg_loss"], 0.1 * np.sum(w * w), rtol=1e-6)
        np.testing.assert_allclose(g, 0.2 * w, rtol=1e-6)
        np.testing.assert_allclose(
            values["optimization_loss"],
            sum(values[k] for k in ["data_loss", "dense_reg_loss", "sparse_reg_loss"]),
            rtol=1e-6,
        )
        assert set(evaluation["losses"]) == set(p.tasks)
        assert kernels or active


def test_disabled_identical_graph_values_and_gradients():
    def execute(cfg):
        with tf.Graph().as_default(), tf.Session() as sess:
            random.seed(42)
            np.random.seed(42)
            tf.set_random_seed(42)
            r = model_fn(inputs(), None, "train", parameters(cfg, vocab()))
            grads = [
                tf.convert_to_tensor(g)
                for g in tf.gradients(r["loss"], tf.trainable_variables())
                if g is not None
            ]
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            return sess.run([r["loss"], *grads])

    cfg = config()
    explicit = deepcopy(cfg)
    explicit.update(
        dense_use_weight_decay=False,
        sparse_use_weight_decay=False,
        sparse_weight_decay=0.0,
        sparse_weight_decay_vars=[],
    )
    for a, b in zip(execute(cfg), execute(explicit)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "change",
    [
        {"optimizer": "adamw", "optimizer_args": {"weight_decay": 0.1}},
        {"sparse_weight_decay": -1},
        {"sparse_weight_decay": float("nan")},
        {"sparse_weight_decay_vars": ["missing"]},
        {"dense_use_weight_decay": False},
        {"sparse_use_weight_decay": False},
        {"dense_use_weight_decay": "true"},
    ],
)
def test_invalid_regularization_config(change):
    cfg = regularized()
    cfg.update(change)
    with pytest.raises(ValueError):
        parameters(cfg)


def test_empty_history_and_real_oov():
    cfg = regularized()
    cfg["dense_use_weight_decay"] = False
    cfg["output_layer"]["regs"] = []
    cfg["input_features"] = {"history": cfg["input_features"]["history"]}
    with tf.Graph().as_default(), tf.Session() as sess:
        params = parameters(cfg, vocab())
        length = tf.placeholder(tf.int32, [2])
        features = inputs()
        features["history"]["length"] = length
        _, _, _, embeddings = embedded_features(features, "train", params)
        term = loss_terms(params, embeddings, [])["sparse_reg_loss"]
        w = tf.trainable_variables()[0]
        grad = tf.convert_to_tensor(tf.gradients(term, w)[0])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(w.assign([[3.0] * 4, [1.0] * 4, [2.0] * 4]))
        empty, empty_grad = sess.run([term, grad], {length: [0, 0]})
        assert empty == 0
        np.testing.assert_array_equal(empty_grad, np.zeros((3, 4)))
        loss, gradient = sess.run([term, grad], {length: [0, 1]})
        np.testing.assert_allclose(loss, 0.2 / 2 * 36)
        np.testing.assert_allclose(gradient[0], [0.6] * 4)


def test_adam_multiple_steps_matches_explicit_occurrence_l2():
    cfg = regularized()
    cfg["dense_use_weight_decay"] = False
    cfg["output_layer"]["regs"] = []
    with tf.Graph().as_default(), tf.Session() as sess:
        p = parameters(cfg, vocab())
        ids = tf.placeholder(tf.int32, [None])
        actual = tf.get_variable("actual", initializer=[[1.0], [2.0], [3.0]])
        reference = tf.get_variable("reference", initializer=[[1.0], [2.0], [3.0]])
        from recommend_tf.train.comm.graph_helper import SPARSE_VARIABLES

        tf.add_to_collection(SPARSE_VARIABLES, actual)
        values = tf.nn.embedding_lookup(actual, ids)
        loss = loss_terms(p, {"drama": [values]}, [])["sparse_reg_loss"]
        update = optimize_loss(loss, p)
        ref_loss = 0.2 / 2 * tf.reduce_sum(tf.square(tf.nn.embedding_lookup(reference, ids)))
        ref_update = tf.train.AdamOptimizer(0.001).minimize(ref_loss, var_list=[reference])
        sess.run(tf.global_variables_initializer())
        for batch in [[0, 0, 1], [2], [1, 1], [0, 2]]:
            sess.run([update, ref_update], {ids: batch})
            a, b = sess.run([actual, reference])
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
