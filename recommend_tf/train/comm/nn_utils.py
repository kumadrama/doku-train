"""Finder current-path loss formulas; no implicit validity mask."""

import tensorflow.compat.v1 as tf

from .common_utils import eval_expr


def get_predictions(head, logits):
    if head["predict_fn"] == "sigmoid":
        return tf.sigmoid(logits)
    if head["predict_fn"] == "expectation":
        probs = tf.nn.softmax(logits, axis=-1)
        bins = tf.range(head["num_outputs"], dtype=tf.float32)
        return tf.reduce_sum(probs * bins, axis=-1) * float(head["soft_label"]["scale"])
    raise ValueError("Unsupported prediction function")


def soft_label(labels, head):
    option = head["soft_label"]
    if option["func"] != "normal":
        raise ValueError("Only Finder normal soft labels are in scope")
    y = tf.cast(labels, tf.float32) / float(option["scale"])
    sigma = float(option["sigma"]) / float(option["scale"])
    dist = tf.distributions.Normal(0.0, sigma)
    ybins = tf.range(head["num_outputs"], dtype=tf.float32)[None, :] - y[:, None]
    softy = dist.prob(ybins)
    return softy / tf.reduce_sum(softy, axis=-1, keepdims=True)


def get_loss(head, logits, labels, features):
    weight = head["loss_weight"]
    if isinstance(weight, str):
        weight = eval_expr(
            weight,
            {
                "logits": logits,
                "labels": labels,
                "predictions": get_predictions(head, logits),
                **features["item"],
                **features,
            },
        )
    if head["loss_type"] == "logloss":
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32), logits=logits
        )
    elif head["loss_type"] == "softmax_soft":
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    else:
        raise ValueError("Unsupported loss type")
    if head.get("loss_by_weight", False):
        loss = tf.cond(
            tf.equal(tf.size(losses), 0),
            lambda: 0.0,
            lambda: (
                tf.reduce_sum(weight * losses)
                / (tf.math.count_nonzero(weight, dtype=tf.float32) + 1e-10)
            ),
        )
    else:
        loss = tf.cond(
            tf.equal(tf.size(losses), 0), lambda: 0.0, lambda: tf.reduce_mean(weight * losses)
        )
    return loss, weight
