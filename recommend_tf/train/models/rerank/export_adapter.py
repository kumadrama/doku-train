"""Narrow model-owned checkpoint export adapter; no training orchestration."""

import hashlib
import random

import numpy as np
import tensorflow.compat.v1 as tf

from .model_params import parameters
from .rerank_model import model_fn


def build(config, vocabs, values, mode):
    # Initializer nodes remain in SavedModel even though checkpoint values are restored.
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    inputs = {
        group: {
            name: tf.placeholder(
                tf.as_dtype(value.dtype), [None] * value.ndim, name=f"{group}/{name}"
            )
            for name, value in fields.items()
        }
        for group, fields in values.items()
    }
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        result = model_fn(inputs, None, mode, parameters(config, vocabs))
    return inputs, result["predictions"]


def feed(inputs, values):
    return {
        tensor: values[group][name]
        for group, fields in inputs.items()
        for name, tensor in fields.items()
    }


def reachable_inputs(outputs, inputs):
    pending = [tensor.op for tensor in outputs.values()]
    seen = set()
    while pending:
        op = pending.pop()
        if op in seen:
            continue
        seen.add(op)
        pending.extend(t.op for t in op.inputs)
        pending.extend(op.control_inputs)
    return {
        f"{group}__{name}": tensor
        for group, fields in inputs.items()
        for name, tensor in fields.items()
        if tensor.op in seen
    }


def export(config, vocabs, checkpoint, output, batches, tasks, forbidden, namespace):
    tf.disable_eager_execution()
    options = tf.ConfigProto(
        device_count={"GPU": 0},
        intra_op_parallelism_threads=config["intra_op_threads"],
        inter_op_parallelism_threads=config["inter_op_threads"],
    )
    references = []
    with tf.Graph().as_default():
        inputs, predictions = build(config["model"], vocabs, batches[0], tf.estimator.ModeKeys.EVAL)
        saver = tf.train.Saver()
        with tf.Session(config=options) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            saver.restore(session, str(checkpoint / "model"))
            for values in batches:
                references.append(
                    session.run({k: predictions[k] for k in tasks}, feed(inputs, values))
                )
    with tf.Graph().as_default():
        inputs, predictions = build(
            config["model"], vocabs, batches[0], tf.estimator.ModeKeys.PREDICT
        )
        outputs = {k: predictions[k] for k in tasks}
        signature_inputs = reachable_inputs(outputs, inputs)
        if any(
            tensor is inputs["item"].get(name)
            for tensor in signature_inputs.values()
            for name in forbidden
        ):
            raise ValueError("prediction depends on a label/audit-only input")
        if any(
            op.type in {"PyFunc", "PyFuncStateless", "EagerPyFunc"}
            for op in tf.get_default_graph().get_operations()
        ):
            raise ValueError("Python callbacks cannot be exported")
        saver = tf.train.Saver()
        with tf.Session(config=options) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            saver.restore(session, str(checkpoint / "model"))
            builder = tf.saved_model.builder.SavedModelBuilder(str(output))
            signature = tf.saved_model.signature_def_utils.predict_signature_def(
                signature_inputs, outputs
            )
            builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={"serving_default": signature},
                main_op=tf.tables_initializer(),
            )
            builder.save()
            schema = {
                "inputs": {
                    k: {"dtype": v.dtype.name, "shape": v.shape.as_list(), "tensor": v.name}
                    for k, v in signature_inputs.items()
                },
                "outputs": {
                    k: {"dtype": v.dtype.name, "shape": v.shape.as_list(), "tensor": v.name}
                    for k, v in outputs.items()
                },
            }
        keys = {
            key: next(
                (g, n) for g, fields in inputs.items() for n, t in fields.items() if t is tensor
            )
            for key, tensor in signature_inputs.items()
        }
    for key, (group, field) in keys.items():
        schema["inputs"][key].update(source_group=group, source_field=field)
    # StaticHashTable generates UUID shared names. Canonicalize only the exported
    # graph, scoped by immutable model version, then verify the rewritten artifact.
    from tensorflow.core.protobuf.saved_model_pb2 import SavedModel

    protobuf = SavedModel()
    protobuf.ParseFromString((output / "saved_model.pb").read_bytes())
    for meta_graph in protobuf.meta_graphs:
        groups = {}
        for node in meta_graph.graph_def.node:
            if node.op in {"HashTableV2", "HashTable"}:
                groups.setdefault(node.attr["shared_name"].s, []).append(node)
        for nodes in groups.values():
            resource = (
                hashlib.sha256((namespace + min(n.name for n in nodes)).encode())
                .hexdigest()
                .encode()
            )
            for node in nodes:
                node.attr["shared_name"].s = resource
    (output / "saved_model.pb").write_bytes(protobuf.SerializeToString(deterministic=True))
    with tf.Graph().as_default(), tf.Session(config=options) as session:
        meta = tf.saved_model.loader.load(
            session, [tf.saved_model.tag_constants.SERVING], str(output)
        )
        signature = meta.signature_def["serving_default"]
        graph = tf.get_default_graph()
        for values, expected in zip(batches, references):
            actual = session.run(
                {k: graph.get_tensor_by_name(v.name) for k, v in signature.outputs.items()},
                {
                    graph.get_tensor_by_name(signature.inputs[k].name): values[g][n]
                    for k, (g, n) in keys.items()
                },
            )
            for name in tasks:
                if not np.isfinite(actual[name]).all():
                    raise ValueError("nonfinite exported predictions")
                np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)
    return schema
