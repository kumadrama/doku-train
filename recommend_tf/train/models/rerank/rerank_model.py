"""Finder rerank current-path migration; see source-inventory.json."""

import ast
from functools import partial

import tensorflow.compat.v1 as tf

from recommend_tf.train.comm import graph_helper
from recommend_tf.train.comm.common_utils import eval_expr
from recommend_tf.train.comm.layernorm import LayerNorm
from recommend_tf.train.models.rerank.modules.utils import fm_layer

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT


class _Debug:
    @staticmethod
    def add_debug_tensor(name, value, *args):
        tf.add_to_collection("finder_debug/" + name, value)


debug_utils = _Debug()


def _unsupported(*args, **kwargs):
    raise ValueError("This layer branch is outside the migrated experiment")


class _NN:
    dice = staticmethod(_unsupported)
    parametric_relu = staticmethod(_unsupported)


nn_utils = _NN()


def _get_norm(x, debug_name, norm, mode):
    if norm == "":
        return lambda x: x
    if norm == "ln":
        with tf.variable_scope(f"{debug_name}_layernorm", reuse=tf.AUTO_REUSE):
            layer_norm = LayerNorm(x, center=True, scale=True, epsilon=1e-6)
            return lambda value: layer_norm.forward(value)
    raise ValueError("Unsupported normalization: " + norm)


def transform_features(features, params):
    features = tf.nest.map_structure(lambda x: x, features)  # deep copy
    for trans in params.feature_transformations:
        try:
            exprs = (
                [trans["expr"].replace("[*]", feat) for feat in trans["apply_to_features"]]
                if len(trans["apply_to_features"]) > 0
                else [trans["expr"]]
            )
            for expr in exprs:
                ast_tree = ast.parse(expr)
                assert len(ast_tree.body) == 1
                expr = ast_tree.body[0]
                assert isinstance(expr, ast.Assign)
                assert len(expr.targets) == 1
                assert isinstance(expr.targets[0], ast.Name)
                for feature_type in trans["apply_to"]:
                    variables = {**features[feature_type], **features}
                    key = expr.targets[0].id
                    val = eval_expr(expr.value, variables)
                    features[feature_type][key] = val
                    if key in params.debug_tensor_histogram:
                        tf.summary.histogram(f"feature_transform/{key}", val)
                    if key in params.debug_sample_age:
                        tf.summary.scalar(f"debug_sample_age/{key}", val)
        except:  # pylint: disable=bare-except
            tf.logging.error("error applying transformation: %s", trans)
            raise
    return features


def dnn_layer(layer, x, mode, debug_name, lhuc_inputs=None):
    act_fn = {
        "relu": tf.nn.relu,
        "dice": partial(nn_utils.dice, mode=mode),
        "prelu": nn_utils.parametric_relu,
        "": lambda x: x,
        "silu": tf.nn.silu,
    }

    dims = layer["dims"]
    acts = layer["acts"]
    regs = layer["regs"]
    use_silu = layer["use_silu"]
    use_film = layer["use_film"]
    dropout = layer["layer_dropout"]
    norms = layer.get("norms", [""] * len(dims))
    assert len(dims) == len(acts)
    assert len(dims) == len(norms) or len(dims) + 1 == len(norms)
    initializer = tf.keras.initializers.VarianceScaling()
    regularizer = tf.keras.regularizers.L1L2(l1=regs[0], l2=regs[1]) if regs else None

    if use_film:
        film_dims = 2 * len(dims)
        with tf.variable_scope(f"{debug_name}_film_layer", reuse=tf.AUTO_REUSE):
            film_params = tf.layers.dense(x, units=film_dims)
            _betas = tf.reshape(film_params, [2, -1])[0]
            _gammas = tf.reshape(film_params, [2, -1])[1]
            B = tf.shape(x)[0]
            betas = tf.split(_betas, [B] * len(dims))
            gammas = tf.split(_gammas, [B] * len(dims))

    for i, (dim, act, norm) in enumerate(zip(dims, acts, norms)):
        if lhuc_inputs is not None:
            with tf.variable_scope(f"{debug_name}_f{i}_lhuc_layer", reuse=tf.AUTO_REUSE):
                lhuc_dim = x.get_shape().as_list()[-1]
                lhuc_gate = tf.layers.dense(
                    lhuc_inputs,
                    units=lhuc_dim,
                    activation=tf.nn.sigmoid,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                )
                x = x * lhuc_gate * 2

        with tf.variable_scope(f"layer{i}", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(
                x,
                units=dim,
                activation=None,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            )

            norm_fn = _get_norm(x, f"{debug_name}_{i}", norm, mode)
            x = norm_fn(x)
            if use_film:
                x = betas[i][:, tf.newaxis] * x + gammas[i][:, tf.newaxis]

            x = act_fn[act](x)
            if use_silu:
                x = x * tf.nn.sigmoid(x)

            if dropout < 1.0:
                x = tf.layers.dropout(x, rate=1 - dropout, training=(mode == TRAIN))
            debug_utils.add_debug_tensor(f"{debug_name}_{i}", x)

    if len(dims) + 1 == len(norms):
        x = norm_fn[norms[-1]](x)
    return x


def target_layer(layer, x, mode, debug_name, lhuc_inputs=None):
    if layer["stop_grad"]:
        x = tf.stop_gradient(x)
        if lhuc_inputs is not None:
            lhuc_inputs = tf.stop_gradient(lhuc_inputs)

    assert layer["num_outputs"] > 0
    x = dnn_layer(layer, x, mode, debug_name, lhuc_inputs)
    if lhuc_inputs is not None:
        with tf.variable_scope(f"{debug_name}_out_lhuc_layer", reuse=tf.AUTO_REUSE):
            lhuc_dim = x.get_shape().as_list()[-1]
            lhuc_gate = tf.layers.dense(
                lhuc_inputs,
                units=lhuc_dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
            )
            x = x * lhuc_gate * 2
    x = tf.layers.dense(x, units=layer["num_outputs"], activation=None)
    if layer["num_outputs"] == 1:
        x = tf.squeeze(x, axis=[-1])
    return x


def scalar_features(features, mode, params):
    outputs = {}
    for name, options in params.input_features.items():
        if options["input_type"] != "scalar":
            continue
        from_feats = features[options["from"]]
        xs = []
        for feat in options["features"]:
            x = from_feats[feat]
            x = tf.cast(x, tf.float32)
            if isinstance(x, tf.SparseTensor):
                x = tf.sparse_tensor_to_dense(x)
            else:
                x = tf.expand_dims(x, axis=1)
            if mode == PREDICT:
                key_from, key_feat = params.broadcast_feat.split("/")
                batch_size = tf.shape(features[key_from][key_feat])[0]
                x = tf.tile(x, [batch_size // tf.shape(x)[0]] + [1] * (x.shape.rank - 1))
            xs.append(x)
        combiner = options.get("combiner", params.embed_combiner)
        combine_fn = {
            "concat": partial(tf.concat, axis=-1),
            "sum": tf.add_n,
        }[combiner]
        outputs[name] = combine_fn(xs)
    return outputs


def embedded_features(features, mode, params):
    from recommend_tf.train.comm.vocab_manager import lookup

    outputs, lengths, sparse_lengths = {}, {}, {}
    ps_embeddings = {}
    for name, options in params.input_features.items():
        if options["input_type"] != "embed":
            continue
        source = features[options["from"]]
        xs = []
        for field in options["features"]:
            table_name = params.embed_name[field]
            valid = (
                tf.sequence_mask(source["length"], tf.shape(source[field])[1])
                if options.get("history_pooling")
                else None
            )
            value = lookup(source[field], table_name, params, valid)
            if params.sparse_use_weight_decay:
                ps_embeddings.setdefault(table_name, []).append(
                    tf.boolean_mask(value, valid) if valid is not None else value
                )
            if options.get("history_pooling"):
                lengths[name] = source["length"]
            xs.append(value)
        if options["combiner"] != "concat":
            raise ValueError("Current path supports concat input groups")
        outputs[name] = tf.concat(xs, axis=-1)
    return outputs, lengths, sparse_lengths, ps_embeddings


def attention_layer(params, mode, inputs, lengths, sparse_lengths, features):
    scores = {}
    for name, options in params.input_features.items():
        pooling = options.get("history_pooling")
        if not pooling:
            continue
        xh = inputs[name]
        mask = tf.sequence_mask(lengths[name], tf.shape(xh)[1], dtype=tf.float32)
        if pooling == "attention":
            xt = inputs[options["hist_attn_query"]][:, tf.newaxis, :]
            score = tf.matmul(xt, xh, transpose_b=True) / params.attn_scale
            score += (1 - mask[:, tf.newaxis, :]) * tf.float32.min
            score = tf.nn.softmax(score)
            # Explicit empty-history adaptation; no attention probability on padding.
            score *= mask[:, tf.newaxis, :]
            inputs[name] = tf.squeeze(tf.matmul(score, xh), axis=1)
            scores[name] = tf.squeeze(score, axis=1)
        elif pooling == "weighted_mean":
            weights = tf.cast(features[options["from"]][options["weight_feature"]], tf.float32)
            weights *= mask
            inputs[name] = tf.math.divide_no_nan(
                tf.reduce_sum(xh * weights[:, :, None], axis=1),
                tf.reduce_sum(weights, axis=1, keepdims=True),
            )
        elif pooling == "mean":
            inputs[name] = tf.math.divide_no_nan(
                tf.reduce_sum(xh * mask[:, :, None], axis=1),
                tf.reduce_sum(mask, axis=1, keepdims=True),
            )
        else:
            raise ValueError("Unsupported history pooling: " + pooling)
    return scores


def rerank_core_model(params, mode, inputs, lengths, sparse_lengths):
    x = tf.concat([inputs[name] for name in params.dnn_inputs], axis=1)
    with tf.variable_scope("output_layers" + params.output_layer_prefix, reuse=tf.AUTO_REUSE):
        x = dnn_layer(params.output_layer, x, mode, "output_layer")
    xs = [x]
    for pair in params.fm_layers:
        fm = fm_layer(params, [inputs[name] for name in pair])
        xs.append(tf.reduce_sum(fm, axis=1, keepdims=True))
    x_new = tf.concat(xs, axis=1)
    model_outputs = {}
    with tf.variable_scope("target_layers" + params.target_layer_prefix, reuse=tf.AUTO_REUSE):
        for task, head in params.tasks.items():
            with tf.variable_scope(params.target_scope_names.get(task, task), reuse=tf.AUTO_REUSE):
                model_outputs[task] = target_layer(head, x_new, mode, f"{task}_output_layer")
                debug_utils.add_debug_tensor(f"{task}_logits", model_outputs[task])
    return model_outputs, x_new


def model_fn(features, labels, mode, params):
    from recommend_tf.train.comm.nn_utils import get_loss, get_predictions, soft_label

    # Capture only regularizers created by this model construction; EVAL reuse must
    # not add TRAIN's collection a second time.
    regularizers_before = {id(x) for x in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)}
    raw_features, features = features, transform_features(features, params)
    scalar_inputs = scalar_features(features, mode, params)
    inputs, lengths, sparse_lengths, ps_embeddings = embedded_features(features, mode, params)
    inputs.update(scalar_inputs)
    attention_layer(params, mode, inputs, lengths, sparse_lengths, features)
    logits, _ = rerank_core_model(params, mode, inputs, lengths, sparse_lengths)
    predictions = {name: get_predictions(head, logits[name]) for name, head in params.tasks.items()}
    result = {"predictions": predictions, "logits": logits}
    if mode == PREDICT:
        return result
    losses, labels, weights = {}, {}, {}
    for name, head in params.tasks.items():
        source = features if head["from"] == "features" else raw_features
        labels[name] = eval_expr(head["label"], source["item"])
        y = soft_label(labels[name], head) if "soft_label" in head else labels[name]
        losses[name], weights[name] = get_loss(head, logits[name], y, raw_features)
    data_loss = tf.add_n(list(losses.values()))
    from recommend_tf.train.comm.regularization import loss_terms

    regularization = (
        loss_terms(
            params,
            ps_embeddings,
            [
                x
                for x in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                if id(x) not in regularizers_before
            ],
        )
        if mode == TRAIN
        else {}
    )
    loss = tf.add_n([data_loss, *regularization.values()]) if regularization else data_loss
    metric_values = {}
    denominators = {}
    for name, head in params.tasks.items():
        source = features if head["from"] == "features" else raw_features
        metric_values[name] = {
            metric: eval_expr(
                option.get("label", head.get("metric_label", head["label"])), source["item"]
            )
            for metric, option in head.get("metrics", {}).items()
        }
        denominators[name] = (
            tf.math.count_nonzero(weights[name], dtype=tf.float32)
            if head.get("loss_by_weight", False)
            else tf.cast(tf.size(labels[name]), tf.float32)
        )
    result.update(
        loss=loss,
        objective_losses=dict(data_loss=data_loss, **regularization, optimization_loss=loss),
        losses=losses,
        labels=labels,
        weights=weights,
        loss_denominators=denominators,
        metric_values=metric_values,
    )
    if mode == TRAIN:
        result["train_op"] = graph_helper.optimize_loss(loss, params)
    return result
