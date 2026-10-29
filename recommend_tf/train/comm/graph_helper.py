"""Local variable adapter for Finder optimizer selection and application."""

import tensorflow.compat.v1 as tf

from .weight_decay_optimizers import AdamWOptimizer

SPARSE_VARIABLES = "finder_local_sparse_variables"


def make_optimizer(name, learning_rate, args):
    if name not in ("adam", "adamw"):
        raise ValueError("Unsupported optimizer: " + name)
    if name == "adamw" and "weight_decay" not in args:
        raise ValueError("Finder AdamW requires explicit weight_decay")
    if name == "adam" and "weight_decay" in args:
        raise ValueError("Adam does not implicitly interpret weight_decay")
    cls = tf.train.AdamOptimizer if name == "adam" else AdamWOptimizer
    return cls(learning_rate=learning_rate, **args)


def optimize_loss(loss, params):
    dense_opt = make_optimizer(params.optimizer, params.learning_rate, params.optimizer_args)
    sparse_opt = (
        make_optimizer(
            params.sparse_optimizer, params.sparse_learning_rate, params.sparse_optimizer_args
        )
        if params.sparse_optimizer
        else dense_opt
    )
    sparse = tf.get_collection(SPARSE_VARIABLES)
    sparse_ids = {id(v) for v in sparse}
    dense = [v for v in tf.trainable_variables() if id(v) not in sparse_ids]
    entries = {}
    for opt, variables in ((dense_opt, dense), (sparse_opt, sparse)):
        for grad, var in zip(tf.gradients(loss, variables), variables):
            if grad is None:
                continue
            if isinstance(grad, tf.IndexedSlices):
                indices, inverse = tf.unique(grad.indices)
                grad = tf.IndexedSlices(
                    tf.math.unsorted_segment_sum(grad.values, inverse, tf.size(indices)),
                    indices,
                    grad.dense_shape,
                )
            entries.setdefault(opt, []).append((grad, var))
    # Finder AdamW creates decay assignments independently of gradient inputs.
    # Freeze every gradient before any optimizer can mutate shared parameters.
    dependencies = [
        tensor
        for gvs in entries.values()
        for grad, _ in gvs
        for tensor in (
            [grad.values, grad.indices] if isinstance(grad, tf.IndexedSlices) else [grad]
        )
    ]
    ready = tf.group(*dependencies, name="all_gradients_ready")
    with tf.control_dependencies([ready]):
        updates = [opt.apply_gradients(gvs) for opt, gvs in entries.items()]
    with tf.control_dependencies(updates):
        return tf.train.get_or_create_global_step().assign_add(1)
