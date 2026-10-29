"""Finder example-model loss regularization, adapted to local lookup tensors."""

import tensorflow.compat.v1 as tf


def loss_terms(params, embeddings, dense_losses):
    result = {}
    if params.dense_use_weight_decay:
        if not dense_losses:
            raise ValueError("Dense L2 enabled but no kernel regularizers were constructed")
        result["dense_reg_loss"] = tf.add_n(dense_losses)
    if params.sparse_use_weight_decay:
        selected = {
            name: values
            for name, values in embeddings.items()
            if not params.sparse_weight_decay_vars
            or any(part in name for part in params.sparse_weight_decay_vars)
        }
        tf.logging.info("Embedding L2 tables: %s", sorted(selected))
        terms = [tf.nn.l2_loss(value) for values in selected.values() for value in values]
        result["sparse_reg_loss"] = params.sparse_weight_decay * (
            tf.add_n(terms) if terms else tf.constant(0.0)
        )
    return result
