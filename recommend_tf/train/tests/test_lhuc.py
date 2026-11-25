"""Finder LHUC contract: independent formula oracle, gradients and disabled parity."""

from copy import deepcopy

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.training_pipeline import seed_graph
from recommend_tf.train.models.rerank import rerank_model as model
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_migration import config
from recommend_tf.train.tests.test_regularization import inputs, vocab


def core_config():
    cfg = config()
    cfg.update(
        dnn_inputs=["item"],
        lhuc_inputs=["scalar", "history"],
        lhuc_share_embed=False,
        target_layer_use_lhuc=True,
        output_layer={"dims": [3, 2], "acts": ["relu", "relu"]},
        tasks={"a": cfg["tasks"]["a"], "watch": cfg["tasks"]["watch"]},
    )
    return cfg


@pytest.mark.parametrize(
    "change,match",
    [
        ({"lhuc_inputs": "item"}, "lhuc_inputs"),
        ({"lhuc_inputs": None}, "lhuc_inputs"),
        ({"lhuc_inputs": [1]}, "lhuc_inputs"),
        ({"lhuc_inputs": [""]}, "lhuc_inputs"),
        ({"lhuc_inputs": [[]]}, "lhuc_inputs"),
        ({"lhuc_inputs": ["item", "item"]}, "unique"),
        ({"lhuc_inputs": ["id"]}, "input_features"),
        ({"lhuc_share_embed": "true"}, "boolean"),
        ({"target_layer_use_lhuc": 1}, "boolean"),
    ],
)
def test_invalid_config(change, match):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError, match=match):
        parameters(cfg)


@pytest.mark.parametrize("target", [False, True])
@pytest.mark.parametrize("fm", [False, True])
def test_core_matches_finder_formula_and_condition_gradient(target, fm):
    cfg = core_config()
    cfg["target_layer_use_lhuc"] = target
    if not fm:
        cfg["fm_layers"] = []
    raw = {
        "item": np.array([[0.2, 0.5], [0.7, 0.3]], np.float32),
        "history": np.array([[0.6, 0.1], [0.4, 0.9]], np.float32),
        "scalar": np.array([[0.3], [0.8]], np.float32),
    }
    with tf.Graph().as_default(), tf.Session() as session:
        tensors = {k: tf.constant(v) for k, v in raw.items()}
        output, trunk = model.rerank_core_model(parameters(cfg), "eval", tensors, {}, {})
        gradient = tf.gradients(tf.reduce_sum(output["a"]), tensors["scalar"])[0]
        # Nonuniform weights make input order, layer width and missing gates observable.
        weights = {
            v.name: np.linspace(0.05, 0.25, v.shape.num_elements(), dtype=np.float32).reshape(
                v.shape.as_list()
            )
            for v in tf.trainable_variables()
        }
        session.run([v.assign(weights[v.name]) for v in tf.trainable_variables()])
        actual, shared, grad = session.run([output, trunk, gradient])

    def reference(values):
        condition = np.concatenate([values[n] for n in cfg["lhuc_inputs"]], axis=1)

        def dense(x, scope):
            return x @ weights[scope + "/dense/kernel:0"] + weights[scope + "/dense/bias:0"]

        def gate(x, scope):
            return x * (2 / (1 + np.exp(-dense(condition, scope))))

        x = values["item"]
        for i in range(2):
            x = gate(x, f"output_layers/output_layer_f{i}_lhuc_layer")
            x = np.maximum(dense(x, f"output_layers/layer{i}"), 0)
        if fm:
            x = np.concatenate(
                [x, np.sum(values["item"] * values["history"], axis=1, keepdims=True)], axis=1
            )
        predictions = {}
        for name in cfg["tasks"]:
            scope = f"target_layers/{name}"
            y = gate(x, f"{scope}/{name}_output_layer_f0_lhuc_layer") if target else x
            y = np.maximum(dense(y, scope + "/layer0"), 0)
            if target:
                y = gate(y, f"{scope}/{name}_output_layer_out_lhuc_layer")
            y = dense(y, scope)
            predictions[name] = y[:, 0] if name == "a" else y
        return predictions, x

    expected, expected_shared = reference(raw)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6)
    np.testing.assert_allclose(shared, expected_shared, rtol=1e-6)
    delta = 0.001
    plus, minus = deepcopy(raw), deepcopy(raw)
    plus["scalar"] += delta
    minus["scalar"] -= delta
    finite_difference = (reference(plus)[0]["a"] - reference(minus)[0]["a"]) / (2 * delta)
    np.testing.assert_allclose(grad[:, 0], finite_difference, rtol=0.003, atol=1e-5)
    assert any("out_lhuc_layer" in name for name in weights) == target


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_gradient_boundaries_preserve_main_path_and_trainable_gates(share, stop):
    cfg = core_config()
    cfg.update(lhuc_share_embed=share, fm_layers=[], lhuc_inputs=["scalar", "item"])
    cfg["tasks"]["a"]["stop_grad"] = stop
    with tf.Graph().as_default(), tf.Session() as session:
        tensors = {"item": tf.ones([2, 2]), "scalar": tf.ones([2, 1])}
        output, _ = model.rerank_core_model(parameters(cfg), "train", tensors, {}, {})
        loss = tf.reduce_sum(output["a"])
        item_grad, condition_grad = tf.gradients(loss, [tensors["item"], tensors["scalar"]])
        assert (item_grad is None) == stop
        assert (condition_grad is None) == (share or stop)
        variables = tf.trainable_variables()
        grads = tf.gradients(loss, variables)
        active = {v.name: g for v, g in zip(variables, grads) if g is not None}
        assert any("target_layers/a/" in n and "lhuc" in n for n in active)
        assert any("output_layers/" in n for n in active) == (not stop)
        assert not any("target_layers/watch/" in n for n in active)
        session.run([v.assign(tf.ones_like(v) * 0.1) for v in variables])
        values = session.run(list(active.values()) + ([item_grad] if item_grad is not None else []))
        assert all(np.isfinite(g).all() and np.any(g != 0) for g in values)


def test_regularization_and_eval_variable_reuse():
    cfg = core_config()
    cfg["dense_use_weight_decay"] = True
    for layer in [cfg["output_layer"], *cfg["tasks"].values()]:
        layer["regs"] = [0.0, 0.1]
    with tf.Graph().as_default(), tf.Session() as session:
        p = parameters(cfg)
        tensors = {"item": tf.ones([2, 2]), "history": tf.ones([2, 2]), "scalar": tf.ones([2, 1])}
        model.rerank_core_model(p, "train", tensors, {}, {})
        terms = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        assert len(terms) == 8  # Four hidden Dense kernels and their four gate kernels.
        variables = tf.trainable_variables()
        grads = tf.gradients(tf.add_n(terms), variables)
        regularized = [v for v, g in zip(variables, grads) if g is not None]
        assert len(regularized) == 8
        assert all(
            v.name.endswith("kernel:0") and "out_lhuc_layer" not in v.name for v in regularized
        )
        model.rerank_core_model(p, "eval", tensors, {}, {})
        assert [v.name for v in tf.trainable_variables()] == [v.name for v in variables]
        assert len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) == 8
        session.run(tf.global_variables_initializer())
        value, kernels = session.run([tf.add_n(terms), regularized])
        np.testing.assert_allclose(value, 0.1 * sum(np.sum(w * w) for w in kernels), rtol=1e-6)


def test_empty_hidden_task_still_gates_logits():
    cfg = core_config()
    cfg["tasks"]["a"].update(dims=[], acts=[])
    with tf.Graph().as_default():
        tensors = {"item": tf.ones([2, 2]), "history": tf.ones([2, 2]), "scalar": tf.ones([2, 1])}
        model.rerank_core_model(parameters(cfg), "eval", tensors, {}, {})
        gates = [v.name for v in tf.trainable_variables() if "target_layers/a/" in v.name]
        assert any("out_lhuc_layer" in n for n in gates)
        assert not any("f0_lhuc_layer" in n for n in gates)


def test_disabled_matches_original_graph_and_one_update(monkeypatch):
    def original_core(p, mode, inputs, lengths, sparse_lengths):
        x = tf.concat([inputs[name] for name in p.dnn_inputs], axis=1)
        with tf.variable_scope("output_layers" + p.output_layer_prefix, reuse=tf.AUTO_REUSE):
            x = model.dnn_layer(p.output_layer, x, mode, "output_layer")
        xs = [x]
        for pair in p.fm_layers:
            xs.append(
                tf.reduce_sum(model.fm_layer(p, [inputs[n] for n in pair]), axis=1, keepdims=True)
            )
        shared = tf.concat(xs, axis=1)
        outputs = {}
        with tf.variable_scope("target_layers" + p.target_layer_prefix, reuse=tf.AUTO_REUSE):
            for name, head in p.tasks.items():
                with tf.variable_scope(p.target_scope_names.get(name, name), reuse=tf.AUTO_REUSE):
                    outputs[name] = model.target_layer(head, shared, mode, f"{name}_output_layer")
                    model.debug_utils.add_debug_tensor(f"{name}_logits", outputs[name])
        return outputs, shared

    def execute(cfg):
        with tf.Graph().as_default(), tf.Session() as session:
            seed_graph(42)
            result = model.model_fn(inputs(), None, "train", parameters(cfg, vocab()))
            variables = tf.trainable_variables()
            gradients = [tf.convert_to_tensor(g) for g in tf.gradients(result["loss"], variables)]
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            before = session.run([result["loss"], *result["predictions"].values(), *gradients])
            session.run(result["train_op"])
            return [v.name for v in variables], before + session.run(variables)

    cfg = config()
    current = execute(cfg)
    empty = execute(dict(cfg, lhuc_inputs=[], lhuc_share_embed=False, target_layer_use_lhuc=True))
    monkeypatch.setattr(model, "rerank_core_model", original_core)
    legacy = execute(cfg)
    assert current[0] == legacy[0] == empty[0]
    for actual in [current, empty]:
        for left, right in zip(actual[1], legacy[1]):
            np.testing.assert_array_equal(left, right)


def test_country_only_gates_ignore_platform_and_country_still_learns_from_main_path():
    cfg = config()
    cfg.update(
        input_features={
            "context": {"features": ["ctx_platform"]},
            "country": {"features": ["ctx_ip_country"]},
        },
        embed_name={"ctx_platform": "platform", "ctx_ip_country": "ip_country"},
        embed_dim={"*": 2},
        dnn_inputs=["context", "country"],
        fm_layers=[],
        lhuc_inputs=["country"],
        lhuc_share_embed=True,
        target_layer_use_lhuc=True,
        tasks={"a": cfg["tasks"]["a"]},
    )
    cfg["output_layer"]["layer_dropout"] = 1.0
    states = {
        "platform": {"dtype": "string", "keys": ["ios", "android"]},
        "ip_country": {"dtype": "string", "keys": ["US", "CA"]},
    }
    with tf.Graph().as_default() as graph, tf.Session() as session:
        features = {
            "item": {
                "ctx_platform": tf.constant(["ios", "android", "ios"]),
                "ctx_ip_country": tf.constant(["US", "US", "CA"]),
                "click": tf.constant([0, 0, 0]),
                "scene": tf.constant([1, 1, 1]),
            }
        }
        result = model.model_fn(features, None, "train", parameters(cfg, states))
        gates = [
            op.outputs[0]
            for op in graph.get_operations()
            if op.type == "Sigmoid" and "_lhuc_layer/" in op.name
        ]
        assert len(gates) == 3  # Shared hidden, task hidden and task logits gates.
        variables = tf.trainable_variables()
        country = next(v for v in variables if v.name == "local_embeddings/ip_country/weights:0")
        assert all(tf.gradients(tf.reduce_sum(gate), country)[0] is None for gate in gates)
        main_gradient = tf.convert_to_tensor(tf.gradients(result["loss"], country)[0])
        for variable in variables:
            if "_lhuc_layer/" in variable.name and variable.name.endswith("kernel:0"):
                assert variable.shape.as_list()[0] == 2  # Country embedding only.
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        session.run([v.assign(tf.ones_like(v) * 0.1) for v in variables])
        session.run(country.assign([[0.0, 0.0], [0.2, 0.2], [0.8, 0.8]]))
        platform = next(v for v in variables if v.name == "local_embeddings/platform/weights:0")
        session.run(platform.assign([[0.0, 0.0], [0.1, 0.1], [0.6, 0.6]]))
        values, gradient = session.run([gates, main_gradient])
        for gate in values:
            np.testing.assert_array_equal(gate[0], gate[1])
            assert np.all(gate[0] != gate[2])
        assert np.any(gradient[1] != 0) and np.any(gradient[2] != 0)
