import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm import graph_helper


def ancestors(op):
    seen = set()
    pending = [op]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(current.control_inputs)
        pending.extend(x.op for x in current.inputs)
    return seen


@pytest.mark.parametrize("separate_sparse", [False, True])
def test_adamw_all_gradients_before_every_update(monkeypatch, separate_sparse):
    optimizers = []
    original = graph_helper.make_optimizer

    def capture(*args):
        result = original(*args)
        optimizers.append(result)
        return result

    monkeypatch.setattr(graph_helper, "make_optimizer", capture)
    params = SimpleNamespace(
        optimizer="adamw",
        learning_rate=0.0,
        optimizer_args={"weight_decay": 0.1},
        sparse_optimizer="adamw" if separate_sparse else "",
        sparse_learning_rate=0.0,
        sparse_optimizer_args={"weight_decay": 0.1},
    )
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        dense = tf.get_variable("dense", initializer=1.0)
        embed = tf.get_variable("embed", initializer=[[2.0], [3.0], [4.0]])
        tf.add_to_collection(graph_helper.SPARSE_VARIABLES, embed)
        loss = tf.square(dense * tf.reduce_sum(tf.nn.embedding_lookup(embed, [0, 0, 1])))
        step = graph_helper.optimize_loss(loss, params)
        ready = graph.get_operation_by_name("all_gradients_ready")
        # Structural regression: covers decay and Adam writes, across both optimizers.
        mutations = [
            op
            for op in graph.get_operations()
            if op.type
            in (
                "AssignSub",
                "AssignSubVariableOp",
                "ScatterAdd",
                "ResourceScatterAdd",
                "ApplyAdam",
                "ResourceApplyAdam",
            )
        ]
        assert mutations
        assert all(ready in ancestors(op) for op in mutations)
        assert len(ready.control_inputs) >= 3  # Dense gradient, sparse values and indices.
        sess.run(tf.global_variables_initializer())
        sess.run(step)
        d, e, m_dense, m_embed = sess.run(
            [dense, embed, optimizers[0].get_slot(dense, "m"), optimizers[-1].get_slot(embed, "m")]
        )
        np.testing.assert_allclose(d, 0.9)
        np.testing.assert_allclose(e[:, 0], [1.8, 2.7, 4.0])
        # Gradients from pre-decay weights: 98, [28, 14, 0], beta1=0.9.
        np.testing.assert_allclose(m_dense, 9.8, rtol=1e-6)
        np.testing.assert_allclose(m_embed[:, 0], [2.8, 1.4, 0], rtol=1e-6)


def test_independent_process_training_and_resume_are_exact(tmp_path):
    worker = Path(__file__).with_name("reproducibility_worker.py")
    root = Path(__file__).resolve().parents[3]
    results = []
    for name, seed, hashseed in [("a", 42, "1"), ("b", 42, "999"), ("c", 43, "2")]:
        target = tmp_path / name
        args = [sys.executable, str(worker), str(target), str(seed)]
        if name == "c":
            args.append("initial_only")
        env = dict(
            os.environ,
            PYTHONPATH=str(root),
            PYTHONHASHSEED=hashseed,
            TF_CPP_MIN_LOG_LEVEL="3",
            TF_NUM_INTRAOP_THREADS="4",
            TF_NUM_INTEROP_THREADS="1",
        )
        result = subprocess.run(
            args, env=env, cwd=root, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
        results.append(json.loads((target / "result.json").read_text()))
    assert results[0] == results[1]
    assert results[0]["initial"] != results[2]["initial"]
