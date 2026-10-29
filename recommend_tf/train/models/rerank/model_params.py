"""Explicit current-path Finder config; reject silently ignored branches."""

import math
from copy import deepcopy
from types import SimpleNamespace


def layer(value, target=False):
    cfg = {
        "dims": [],
        "acts": [],
        "norms": [],
        "regs": [],
        "use_silu": False,
        "use_film": False,
        "layer_dropout": 1.0,
        "stop_grad": False,
    }
    allowed = set(cfg) | (
        {
            "num_outputs",
            "label",
            "from",
            "loss_type",
            "loss_weight",
            "loss_by_weight",
            "predict_fn",
            "soft_label",
            "metrics",
            "metric_label",
        }
        if target
        else set()
    )
    if set(value) - allowed:
        raise ValueError("Unsupported layer settings: " + str(set(value) - allowed))
    cfg.update(value)
    if len(cfg["dims"]) != len(cfg["acts"]) or any(a != "relu" for a in cfg["acts"]):
        raise ValueError("Current path requires one ReLU per dense layer")
    if cfg["use_film"] or cfg["use_silu"]:
        raise ValueError("Current path does not enable FiLM/SiLU")
    if not isinstance(cfg["regs"], list):
        raise ValueError("regs must be a list")
    if cfg["regs"] and (
        len(cfg["regs"]) != 2
        or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in cfg["regs"]
        )
        or cfg["regs"][0] != 0
    ):
        raise ValueError("regs must be [] or [0.0, finite nonnegative L2]")
    if not 0 < cfg["layer_dropout"] <= 1:
        raise ValueError("layer_dropout is a keep probability")
    if not cfg["norms"]:
        cfg["norms"] = [""] * len(cfg["dims"])
    if len(cfg["norms"]) != len(cfg["dims"]) or any(
        norm not in ("", "ln") for norm in cfg["norms"]
    ):
        raise ValueError("Expected one supported norm per hidden layer")
    if target:
        cfg.setdefault("from", "features")
        cfg.setdefault("loss_weight", "1.0")
        cfg.setdefault("loss_by_weight", False)
        if cfg["num_outputs"] < 1:
            raise ValueError("num_outputs must be positive")
    return cfg


def parameters(config, vocabs=None):
    cfg = deepcopy(config)
    known = {
        "dense_use_weight_decay",
        "sparse_use_weight_decay",
        "sparse_weight_decay",
        "sparse_weight_decay_vars",
        "input_features",
        "feature_transformations",
        "embed_name",
        "embed_dim",
        "embed_initializer_args",
        "dnn_inputs",
        "fm_layers",
        "output_layer",
        "tasks",
        "optimizer",
        "learning_rate",
        "optimizer_args",
        "sparse_optimizer",
        "sparse_learning_rate",
        "sparse_optimizer_args",
        "attn_scale",
        "output_layer_prefix",
        "target_layer_prefix",
        "target_scope_names",
        "broadcast_feat",
        "debug_tensor_histogram",
        "debug_sample_age",
        "embed_combiner",
    }
    unknown = set(cfg) - known
    if unknown:
        raise ValueError("Unsupported model settings: " + str(unknown))
    defaults = dict(
        dense_use_weight_decay=False,
        sparse_use_weight_decay=False,
        sparse_weight_decay=0.0,
        sparse_weight_decay_vars=[],
        attn_scale=1.0,
        output_layer_prefix="",
        target_layer_prefix="",
        target_scope_names={},
        broadcast_feat="item/i_drama_id",
        debug_tensor_histogram=[],
        debug_sample_age=[],
        embed_combiner="concat",
        optimizer="adam",
        learning_rate=0.001,
        optimizer_args={},
        sparse_optimizer="",
        sparse_learning_rate=0.001,
        sparse_optimizer_args={},
        fm_layers=[],
        feature_transformations=[],
        embed_initializer_args={"minval": -0.01, "maxval": 0.01},
    )
    defaults.update(cfg)
    cfg = defaults
    for trans in cfg["feature_transformations"]:
        trans.setdefault("apply_to_features", [])
        trans.setdefault("apply_to", ["item"])
    for group in cfg["input_features"].values():
        group.setdefault("input_type", "embed")
        group.setdefault("combiner", "concat")
        group.setdefault("from", "item")
        if group["input_type"] not in ("scalar", "embed") or group["combiner"] != "concat":
            raise ValueError("Only scalar/embed concat groups are supported")
    cfg["output_layer"] = layer(cfg["output_layer"])
    cfg["tasks"] = {name: layer(head, True) for name, head in cfg["tasks"].items()}
    for key in ("dense_use_weight_decay", "sparse_use_weight_decay"):
        if not isinstance(cfg[key], bool):
            raise ValueError(key + " must be boolean")
    coefficient = cfg["sparse_weight_decay"]
    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(coefficient)
        or coefficient < 0
    ):
        raise ValueError("sparse_weight_decay must be finite and nonnegative")
    names = cfg["sparse_weight_decay_vars"]
    if not isinstance(names, list) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("sparse_weight_decay_vars must be a list of nonempty substrings")
    layers = [cfg["output_layer"], *cfg["tasks"].values()]
    configured = any(x["regs"] and x["regs"][1] > 0 and x["dims"] for x in layers)
    if cfg["dense_use_weight_decay"] != configured:
        raise ValueError(
            "Dense L2 requires both its enable flag and a positive hidden-layer regs L2"
        )
    if not cfg["sparse_use_weight_decay"] and (coefficient or names):
        raise ValueError("Sparse L2 settings require sparse_use_weight_decay")
    if cfg["dense_use_weight_decay"] or cfg["sparse_use_weight_decay"]:
        if cfg["optimizer"] != "adam" or cfg["sparse_optimizer"] not in ("", "adam"):
            raise ValueError("Loss regularization currently requires Adam for dense and sparse")
    if cfg["sparse_use_weight_decay"]:
        tables = {
            cfg["embed_name"][field]
            for group in cfg["input_features"].values()
            if group["input_type"] == "embed"
            for field in group["features"]
        }
        if not any(not names or any(n in table for n in names) for table in tables):
            raise ValueError("Sparse L2 selection matches no input embedding table")
    cfg["vocabs"] = vocabs or {}
    return SimpleNamespace(**cfg)
