"""Model-owned semantic validation for history attention after feature schema fitting."""

from __future__ import annotations

import math
from typing import Any


def attention_inputs(
    config: dict[str, Any], state: dict[str, Any]
) -> dict[str, tuple[str, str, float]]:
    result = {}
    groups = config["input_features"]
    enabled = {
        name: group
        for name, group in groups.items()
        if not group.get("skip_hist_attn", "hist_attn_query" not in group)
    }
    for name, options in enabled.items():
        query = options.get("hist_attn_query")
        if not isinstance(query, str) or query not in groups:
            raise ValueError("attention references unknown input group")
        if name == query or query in enabled:
            raise ValueError("attention query must be a non-attention scalar group")
        fields = []
        dims = []
        for group_name, is_history in [(name, True), (query, False)]:
            group = groups[group_name]
            if len(group["features"]) != 1 or "weight_feature" in group:
                raise ValueError("attention requires one unweighted embedding per group")
            if group.get("combiner") == "weighted_mean":
                raise ValueError("attention cannot combine with weighted_mean")
            field = group["features"][0]
            if field not in state["tables"]:
                raise ValueError("attention requires categorical embeddings")
            dtype = state["dtypes"].get(field, "")
            if not dtype or dtype.startswith("list") != is_history:
                raise ValueError("attention requires a history sequence and scalar query")
            table = state["tables"][field]
            dim = config["embed_dim"].get(table, config["embed_dim"].get("*"))
            if dim is None:
                raise ValueError(f"missing attention embedding dimension: {table}")
            fields.append(field)
            dims.append(dim)
        if dims[0] != dims[1]:
            raise ValueError("attention query and history dimensions differ")
        scale = config.get("attn_scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("attn_scale must be a finite positive number")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attn_scale must be a finite positive number")
        result[name] = (fields[0], fields[1], float(scale))
    return result
