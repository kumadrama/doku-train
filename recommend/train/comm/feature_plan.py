"""Shared, train-fitted flat features and history processing."""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from recommend.train.comm.expressions import (
    compile_expression,
    dependencies,
    evaluate,
    transformations,
)


class FeaturePlan:
    def __init__(self, config: dict[str, Any], dtypes: dict[str, str]) -> None:
        self.config = config
        self.dtypes = dict(dtypes)
        self.transforms = transformations(config)
        self.vocabs: dict[str, dict[Any, int]] = {}
        self.tables = config.get("embed_name", {}) or {
            f: f for f in config["tree_inputs"]["categorical"]
        }
        # Derived embedding fields declare their scalar type in the expression,
        # rather than requiring a fictitious source-Parquet column.
        for target, expr in self.transforms:
            node = compile_expression(expr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "cast"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value in {"int32", "int64", "float32", "float64", "bool"}
            ):
                self.dtypes[target] = node.args[1].value
            elif isinstance(node, ast.Name) and node.id in self.dtypes:
                self.dtypes[target] = self.dtypes[node.id]
        for field in self.tables:
            if field not in self.dtypes:
                raise ValueError(f"embedding field requires a declared cast type: {field}")
        self.lengths: dict[str, int] = {}
        self.history_capacities: dict[str, int] = {}
        policies: dict[str, dict[str, Any]] = {}
        for name, options in config["hist_options"].items():
            for field in config["input_features"][name]["features"]:
                if field in policies and policies[field] != options:
                    raise ValueError("shared history field has conflicting truncation policies")
                policies[field] = options
                self.history_capacities[field] = options["max_hist_size"]
        self.columns = set(config["features"]) | dependencies(config["item_filter"])
        produced: set[str] = set()
        for target, expr in self.transforms:
            self.columns |= dependencies(expr) - produced
            produced.add(target)
        for task in config["tasks"].values():
            for key in ["label", "mask", "loss_weight", "metric_label"]:
                if key in task:
                    self.columns |= dependencies(task[key]) - produced
            for metric in task["metrics"].values():
                for key in ["filter", "label"]:
                    if key in metric:
                        self.columns |= dependencies(metric[key]) - produced

    def row(self, source: dict[str, Any]) -> dict[str, Any] | None:
        row = dict(source)
        if not evaluate(self.config["item_filter"], row):
            return None
        for group in self.config["input_features"].values():
            if "weight_feature" in group:
                ids, weights = row[group["features"][0]], row[group["weight_feature"]]
                if (ids is None) != (weights is None) or (
                    ids is not None and len(ids) != len(weights)
                ):
                    raise ValueError("history ID/weight mismatch")
        for field, default in self.config["feature_default_values"].items():
            if row[field] is None:
                row[field] = default
        for field in self.config["features"]:
            v = row[field]
            if (
                v is not None
                and self.dtypes[field].startswith("list")
                and any(x is None for x in v)
            ):
                raise ValueError("NULL history elements are unsupported")
        for group in self.config["input_features"].values():
            fields = group["features"]
            if len(fields) > 1 and all(self.dtypes.get(f, "").startswith("list") for f in fields):
                lengths = [len(row[f]) if row[f] is not None else 0 for f in fields]
                if len(set(lengths)) != 1:
                    raise ValueError("history fields must have aligned lengths")
        for name, opts in self.config["hist_options"].items():
            group = self.config["input_features"][name]
            field = group["features"][0]
            ids = row[field]
            if ids is None:
                continue
            if opts["truncate"] == "tail":
                if opts["duplicates"] != "preserve":
                    raise ValueError("tail history requires preserve policy")
                for history_field in group["features"]:
                    values = row[history_field]
                    row[history_field] = (
                        values[-opts["max_hist_size"] :] if values is not None else []
                    )
                if "weight_feature" in group:
                    weights = row[group["weight_feature"]]
                    row[group["weight_feature"]] = (
                        weights[-opts["max_hist_size"] :] if weights is not None else []
                    )
            else:
                if len(group["features"]) != 1:
                    raise ValueError("top_weight requires a single history field")
                if (
                    opts["duplicates"] != "sum_weights"
                    or opts.get("tie_break") != "raw_id_ascending"
                ):
                    raise ValueError("top_weight requires sum_weights and stable ID ties")
                weight_field = group["weight_feature"]
                weights = row[weight_field]
                if any(not math.isfinite(w) or w < 0 for w in weights):
                    raise ValueError("invalid history weight")
                scale = max(weights, default=0) or 1
                merged: dict[Any, float] = {}
                for i, w in zip(ids, weights, strict=True):
                    merged[i] = merged.get(i, 0) + float(w) / scale
                pairs = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[: opts["max_hist_size"]]
                total = sum(w for _, w in pairs)
                row[field] = [i for i, _ in pairs]
                row[weight_field] = [w / total if total else 0.0 for _, w in pairs]
        for target, expr in self.transforms:
            row[target] = evaluate(expr, row)
        for field, op in self.config.get("history_scalars", {}).items():
            func = op["func"]
            if func == "unique_intersection_count":
                a, b = row[op["left"]], row[op["right"]]
                row[field] = None if a is None or b is None else len(set(a) & set(b))
                continue
            group = self.config["input_features"][op["history"]]
            ids = row[group["features"][0]]
            candidate = row.get(op.get("candidate", ""))
            if ids is None:
                row[field] = None
            elif func == "count":
                row[field] = len(ids)
            elif func == "contains":
                row[field] = None if candidate is None else int(candidate in ids)
            elif func == "positive_unique_count":
                row[field] = sum(w > 0 for w in row[group["weight_feature"]])
            elif func == "normalized_weight":
                row[field] = (
                    None
                    if candidate is None
                    else sum(
                        w
                        for i, w in zip(ids, row[group["weight_feature"]], strict=True)
                        if i == candidate
                    )
                )
            else:
                raise ValueError("unknown history scalar")
        return row

    def fit(self, rows: Iterator[dict[str, Any]]) -> None:
        values: dict[str, set[Any]] = {table: set() for table in self.tables.values()}
        for source in rows:
            row = self.row(source)
            if row is None:
                continue
            for field, table in self.tables.items():
                v = row[field]
                seq = v if isinstance(v, list) else [] if v is None else [v]
                values[table].update(seq)
                if self.dtypes.get(field, "").startswith("list"):
                    self.lengths[field] = max(self.lengths.get(field, 1), len(seq))
        self.vocabs = {
            table: {v: i + 2 for i, v in enumerate(sorted(vals))} for table, vals in values.items()
        }

    def state(self) -> dict[str, Any]:
        return {
            "version": "feature-state/v2",
            "tables": self.tables,
            "vocabs": {k: list(v.keys()) for k, v in self.vocabs.items()},
            "lengths": self.lengths,
            "padding_index": 0,
            "oov_index": 1,
            "dtypes": self.dtypes,
        }

    def restore(self, state: dict[str, Any]) -> None:
        if state["tables"] != self.tables:
            raise ValueError("checkpoint feature table mismatch")
        self.vocabs = {
            k: {v: i + 2 for i, v in enumerate(vals)} for k, vals in state["vocabs"].items()
        }
        self.lengths = state["lengths"]

    def encode(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        out = {}
        if "tree_inputs" in self.config:
            values = []
            for f in self.config["tree_inputs"]["numeric"]:
                values.append(np.nan if row[f] is None else row[f])
            for f in self.config["tree_inputs"]["categorical"]:
                values.append(
                    np.nan if row[f] is None else self.vocabs[self.tables[f]].get(row[f], 1)
                )
            out["tree"] = np.asarray(values, dtype=np.float32)
        else:
            for name, g in self.config["input_features"].items():
                chunks = []
                for f in g["features"]:
                    value = row[f]
                    if f in self.tables:
                        table = self.vocabs[self.tables[f]]
                        if self.dtypes[f].startswith("list"):
                            seq = [] if value is None else value
                            cap = self.history_capacities.get(
                                f, max(self.lengths.get(f, 1), len(seq))
                            )
                            if len(seq) > cap:
                                raise ValueError("sequence exceeds configured capacity")
                            vector = np.zeros(cap, dtype=np.int64)
                            vector[: len(seq)] = [table.get(v, 1) for v in seq]
                            out[f] = vector
                            if "weight_feature" in g:
                                w = np.zeros(cap, dtype=np.float32)
                                w[: len(seq)] = row[g["weight_feature"]]
                                out[g["weight_feature"]] = w
                        else:
                            out[f] = np.asarray([table.get(value, 1)], dtype=np.int64)
                    else:
                        chunks.append(row[f])
                if chunks:
                    vector = np.asarray(chunks, dtype=np.float32)
                    if not np.isfinite(vector).all():
                        raise ValueError(f"unresolved non-finite numerical group: {name}")
                    out["numeric:" + name] = vector
        for name, t in self.config["tasks"].items():
            mask = bool(evaluate(t["mask"], row))
            label = evaluate(t["label"], row) if mask else 0.0
            metric = evaluate(t.get("metric_label", t["label"]), row) if mask else 0.0
            weight = evaluate(t["loss_weight"], row) if mask else 0.0
            if mask and (label is None or metric is None or weight is None or weight < 0):
                raise ValueError(f"invalid label or weight: {name}")
            out["label:" + name] = np.asarray([label], dtype=np.float32)
            out["metric:" + name] = np.asarray([metric], dtype=np.float32)
            out["mask:" + name] = np.asarray([mask], dtype=np.bool_)
            out["weight:" + name] = np.asarray([weight], dtype=np.float32)
            for key, m in t["metrics"].items():
                valid = mask and bool(evaluate(m.get("filter", "True"), row))
                out["valid:" + name + ":" + key] = np.asarray([valid], dtype=np.bool_)
                if "label" in m:
                    out["value:" + name + ":" + key] = np.asarray(
                        [evaluate(m["label"], row) if valid else 0], dtype=np.float32
                    )
        return out


def prepare_arrays(
    plan: FeaturePlan, rows: Iterator[dict[str, Any]], root: Path, capacity: int, budget: int
) -> dict[str, np.ndarray]:
    root.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    count = 0
    for source in rows:
        row = plan.row(source)
        if row is None:
            continue
        encoded = plan.encode(row)
        if not arrays:
            needed = sum(x.nbytes * capacity for x in encoded.values())
            if needed > budget:
                raise ValueError("encoded dataset exceeds disk budget")
            for i, (key, v) in enumerate(encoded.items()):
                arrays[key] = np.lib.format.open_memmap(
                    root / f"{i}.npy", mode="w+", dtype=v.dtype, shape=(capacity, *v.shape)
                )
        for k, v in encoded.items():
            if arrays[k].shape[1:] != v.shape:
                raise ValueError("feature shape changed across rows")
            arrays[k][count] = v
        count += 1
    if not count:
        raise ValueError("split contains no samples after item_filter")
    return {k: v[:count] for k, v in arrays.items()}
