"""Model-owned field-combination and history-reduction contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroupPlan:
    width: int
    history: bool
    pooling: str | None
    query: str | None


def group_plans(config: dict[str, Any], state: dict[str, Any]) -> dict[str, GroupPlan]:
    groups = config["input_features"]
    plans = {}
    for name, options in groups.items():
        dims = []
        histories = []
        for field in options["features"]:
            dtype = state["dtypes"].get(field, "")
            history = dtype.startswith("list")
            histories.append(history)
            if field in state["tables"]:
                table = state["tables"][field]
                dim = config["embed_dim"].get(table, config["embed_dim"].get("*"))
                if dim is None:
                    raise ValueError(f"missing embedding dimension: {table}")
                dims.append(dim)
            else:
                if history:
                    raise ValueError("history fields must be categorical embeddings")
                dims.append(1)
        if any(histories) != all(histories):
            raise ValueError("cannot combine scalar and history fields in one group")
        combiner = options.get("combiner", "concat")
        if combiner not in {"concat", "sum", "avg"}:
            raise ValueError("combiner only combines fields: concat, sum or avg")
        if combiner != "concat" and len(set(dims)) != 1:
            raise ValueError("sum/avg field combination requires equal dimensions")
        width = sum(dims) if combiner == "concat" else dims[0]
        history = all(histories)
        pooling = options.get("history_pooling")
        if history:
            if pooling not in {"mean", "weighted_mean", "attention", "none"}:
                raise ValueError("history group requires explicit history_pooling")
        elif pooling is not None:
            raise ValueError("scalar group cannot declare history_pooling")
        if (pooling == "weighted_mean") != ("weight_feature" in options):
            raise ValueError("weight_feature requires weighted_mean history_pooling")
        query = options.get("hist_attn_query")
        if (pooling == "attention") != (query is not None):
            raise ValueError("attention pooling requires hist_attn_query exclusively")
        plans[name] = GroupPlan(width, history, pooling, query)
    for plan in plans.values():
        if plan.query is not None:
            if plan.query not in plans:
                raise ValueError("attention references unknown query group")
            query = plans[plan.query]
            if query.history:
                raise ValueError("attention query must be a scalar group")
            if query.width != plan.width:
                raise ValueError("attention query and history dimensions differ after combination")
    consumed = [*config["dnn_inputs"]]
    consumed.extend(n for layer in config["fm_layers"] for pair in layer["features"] for n in pair)
    for name in consumed:
        if name not in plans:
            raise ValueError("unknown model input group")
        if plans[name].history and plans[name].pooling == "none":
            raise ValueError("unreduced history cannot feed DNN or FM")
    scale = config.get("attn_scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("attn_scale must be a finite positive number")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attn_scale must be a finite positive number")
    return plans
