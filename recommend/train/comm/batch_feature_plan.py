"""Bounded column preparation for neural models; legacy scalar path remains the oracle."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from recommend.train.comm import timing
from recommend.train.comm.batch_expressions import Columns, compile_batch_expression, objects
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.comm.runtime_budget import check_memory


class BatchFeaturePlan(FeaturePlan):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        sources = [self.config["item_filter"], *(expr for _, expr in self.transforms), "True"]
        for task in self.config["tasks"].values():
            sources.extend(
                task[k] for k in ("mask", "label", "metric_label", "loss_weight") if k in task
            )
            for metric in task["metrics"].values():
                sources.extend(metric[k] for k in ("filter", "label") if k in metric)
        self.expressions = {expr: compile_batch_expression(expr) for expr in sources}
        self.history_fields = {
            f for f in self.config["features"] if self.dtypes[f].startswith("list")
        }
        for group in self.config["input_features"].values():
            if "weight_feature" in group:
                self.history_fields.update([group["features"][0], group["weight_feature"]])
        self.fitted_dataset: Any = None
        self.widths: dict[str, int] = {}

    def fit(self, rows: Iterator[dict[str, Any]]) -> None:
        # A row iterator cannot establish that the same frozen dataset was scanned.
        # Invalidate before fitting, including when the replacement fit fails.
        self.fitted_dataset = None
        self.widths = {}
        super().fit(rows)

    def restore(self, state: dict[str, Any]) -> None:
        self.fitted_dataset = None
        self.widths = {}
        super().restore(state)

    def expression(self, expr: str, cols: Columns, ids: np.ndarray) -> np.ndarray:
        return self.expressions[expr](cols, ids, self.tree_thresholds)

    def transform_batch(self, source: dict[str, list[Any]]) -> Columns:
        with timing.current().span(
            "batch_transform", count=len(next(iter(source.values()))), unit="rows"
        ):
            check_memory()
            cols = {k: objects(v) for k, v in source.items()}
            ids = np.arange(len(next(iter(cols.values()))))
            keep = self.expression(self.config["item_filter"], cols, ids).astype(bool)
            cols = {k: v[keep] for k, v in cols.items()}
            size = int(keep.sum())
            # Only ragged history columns become row dictionaries. All arithmetic
            # columns stay column-oriented; old history validation/order is reused.
            with timing.current().span("history_rows", count=size, unit="rows"):
                for i in range(size):
                    row = {k: cols[k][i] for k in self.history_fields}
                    self.prepare_history(row, history_only=True)
                    for k, v in row.items():
                        cols[k][i] = v
            for field, default in self.config["feature_default_values"].items():
                if field not in self.history_fields:
                    values = cols[field]
                    missing = np.fromiter((x is None for x in values), dtype=bool, count=size)
                    values[missing] = default
            ids = np.arange(size)
            for target, expr in self.transforms:
                cols[target] = self.expression(expr, cols, ids)
            if self.config.get("history_scalars"):
                with timing.current().span("history_scalar_rows", count=size, unit="rows"):
                    outputs: dict[str, list[Any]] = {k: [] for k in self.config["history_scalars"]}
                    for i in range(size):
                        row = {k: v[i] for k, v in cols.items()}
                        self.history_scalars(row)
                        for k in outputs:
                            outputs[k].append(row[k])
                    cols.update({k: objects(v) for k, v in outputs.items()})
            check_memory()
            return cols

    def fit_dataset(self, dataset: Any) -> None:
        self.fitted_dataset = None
        self.widths = {}
        values: dict[str, set[Any]] = {table: set() for table in self.tables.values()}
        if self.tree_targets and self.tree_thresholds is None:
            raise ValueError("tree bucket artifact is required before fitting features")
        for target, field in self.tree_targets.items():
            if target in self.tables:
                assert self.tree_thresholds is not None
                values[self.tables[target]].update(range(len(self.tree_thresholds[field]) + 1))
        for source in dataset.iter_batches("train", self.columns):
            cols = self.transform_batch(source)
            with timing.current().span(
                "vocab_collect", count=len(next(iter(cols.values()))), unit="rows"
            ):
                for field, table in self.tables.items():
                    for v in cols[field]:
                        seq = v if isinstance(v, list) else [] if v is None else [v]
                        values[table].update(seq)
                        if self.dtypes[field].startswith("list"):
                            self.lengths[field] = max(self.lengths.get(field, 1), len(seq))
                        if isinstance(v, list):
                            self.widths[field] = max(self.widths.get(field, 1), len(v))
        with timing.current().span("vocab_finalize"):
            self.vocabs = {
                table: {v: i + 2 for i, v in enumerate(sorted(vals))}
                for table, vals in values.items()
            }
        self.fitted_dataset = dataset

    def encode_batch(self, cols: Columns) -> dict[str, np.ndarray]:
        size = len(next(iter(cols.values())))
        ids = np.arange(size)
        out: dict[str, np.ndarray] = {}
        for name, group in self.config["input_features"].items():
            numeric = []
            for field in group["features"]:
                if field not in self.tables:
                    numeric.append(field)
                    continue
                table = self.vocabs[self.tables[field]]
                if not self.dtypes[field].startswith("list"):
                    out[field] = np.asarray(
                        [table.get(v, 1) for v in cols[field]], dtype=np.int64
                    ).reshape(size, 1)
                    continue
                cap = self.history_capacities.get(field, self.lengths.get(field, 1))
                check_memory(size * cap * (12 if "weight_feature" in group else 8))
                vector = np.zeros((size, cap), dtype=np.int64)
                weights = (
                    np.zeros((size, cap), dtype=np.float32) if "weight_feature" in group else None
                )
                for i, value in enumerate(cols[field]):
                    seq = [] if value is None else value
                    if len(seq) > cap:
                        raise ValueError("sequence exceeds configured capacity")
                    vector[i, : len(seq)] = [table.get(v, 1) for v in seq]
                    if weights is not None:
                        weights[i, : len(seq)] = cols[group["weight_feature"]][i]
                out[field] = vector
                if weights is not None:
                    out[group["weight_feature"]] = weights
            if numeric:
                vector = np.moveaxis(
                    np.asarray([cols[f].tolist() for f in numeric], dtype=np.float32), 0, 1
                )
                if not np.isfinite(vector).all():
                    raise ValueError(f"unresolved non-finite numerical group: {name}")
                out["numeric:" + name] = vector
        for name, task in self.config["tasks"].items():
            mask = self.expression(task["mask"], cols, ids).astype(bool)
            active = ids[mask]
            labels = self.expression(task["label"], cols, active)
            metrics = self.expression(task.get("metric_label", task["label"]), cols, active)
            weights = self.expression(task["loss_weight"], cols, active)
            if any(
                a is None or b is None or c is None or c < 0
                for a, b, c in zip(labels, metrics, weights, strict=True)
            ):
                raise ValueError(f"invalid label or weight: {name}")
            for prefix, values in [("label", labels), ("metric", metrics), ("weight", weights)]:
                vector = np.zeros((size, 1), dtype=np.float32)
                vector[mask, 0] = values
                if prefix == "weight":
                    out["mask:" + name] = mask.reshape(size, 1)
                out[prefix + ":" + name] = vector
            for key, metric in task["metrics"].items():
                valid = np.zeros(size, dtype=bool)
                valid[mask] = self.expression(metric.get("filter", "True"), cols, active).astype(
                    bool
                )
                out["valid:" + name + ":" + key] = valid.reshape(size, 1)
                if "label" in metric:
                    vector = np.zeros((size, 1), dtype=np.float32)
                    vector[valid, 0] = self.expression(metric["label"], cols, ids[valid])
                    out["value:" + name + ":" + key] = vector
        return out

    def prepare_dataset(self, dataset: Any, split: str, root: Path) -> dict[str, np.ndarray]:
        if split == "train" and dataset is self.fitted_dataset:
            for field, width in self.widths.items():
                self.lengths[field] = max(self.lengths.get(field, 1), width)
        else:
            with timing.current().span("width_scan", report=True):
                for source in dataset.iter_batches(split, self.columns):
                    cols = self.transform_batch(source)
                    for field in self.tables:
                        for value in cols[field]:
                            if isinstance(value, list):
                                self.lengths[field] = max(self.lengths.get(field, 1), len(value))
        root.mkdir(parents=True, exist_ok=False)
        capacity = sum(p["rows"] for p in dataset.partitions(split))
        budget = max(
            0,
            dataset.config["execution"]["cache_limit_bytes"]
            - sum(p.stat().st_size for p in root.parent.rglob("*.npy")),
        )
        arrays: dict[str, np.ndarray] = {}
        count = 0
        for source in dataset.iter_batches(split, self.columns):
            cols = self.transform_batch(source)
            size = len(next(iter(cols.values())))
            if not size:
                continue
            with timing.current().span("batch_encode", count=size, unit="rows"):
                encoded = self.encode_batch(cols)
            check_memory(sum(v.nbytes for v in encoded.values()))
            if not arrays:
                with timing.current().span("array_allocate"):
                    if sum(v[0].nbytes * capacity for v in encoded.values()) > budget:
                        raise ValueError("encoded dataset exceeds disk budget")
                    for i, (key, v) in enumerate(encoded.items()):
                        arrays[key] = np.lib.format.open_memmap(
                            root / f"{i}.npy",
                            mode="w+",
                            dtype=v.dtype,
                            shape=(capacity, *v.shape[1:]),
                        )
            with timing.current().span("array_assign", count=size, unit="rows"):
                for key, v in encoded.items():
                    if arrays[key].shape[1:] != v.shape[1:]:
                        raise ValueError("feature shape changed across rows")
                    arrays[key][count : count + size] = v
            count += size
        if not count:
            raise ValueError("split contains no samples after item_filter")
        return {k: v[:count] for k, v in arrays.items()}
