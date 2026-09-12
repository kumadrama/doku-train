from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa
import torch

from recommend.train.comm.model_params import StrictModel
from recommend.train.models.rerank.data_utils import RerankBatch, completion_labels
from recommend.train.models.rerank.model_params import TARGETS


class NumericFeatureState(StrictModel):
    name: str
    mean: float
    stddev: float
    valid_count: int


class CategoricalFeatureState(StrictModel):
    name: str
    strategy: str = "vocabulary"
    vocabulary: tuple[str, ...]
    missing_index: int = 0
    oov_index: int = 1

    @property
    def cardinality(self) -> int:
        return len(self.vocabulary) + 2


class FittedFeatureState(StrictModel):
    state_version: str = "1"
    feature_schema_version: str
    numeric: tuple[NumericFeatureState, ...]
    categorical: tuple[CategoricalFeatureState, ...]

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @classmethod
    def from_json_bytes(cls, body: bytes) -> FittedFeatureState:
        return cls.model_validate_json(body)


@dataclass(slots=True)
class _NumericAccumulator:
    count: int = 0
    mean: float = 0.0
    squared_deviation_sum: float = 0.0

    def add(self, value: float, *, feature_name: str) -> None:
        if not math.isfinite(value):
            raise ValueError(f"NUMERICAL_FAILURE: non-finite feature {feature_name}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_deviation_sum += delta * (value - self.mean)


def _require_columns(batch: pa.RecordBatch, columns: Iterable[str]) -> None:
    missing = set(columns) - set(batch.schema.names)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")


def fit_feature_state(
    batches: Iterable[pa.RecordBatch],
    *,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    feature_schema_version: str,
    max_vocabulary_size: int = 10_000,
) -> FittedFeatureState:
    numeric_statistics = {name: _NumericAccumulator() for name in numeric_features}
    categorical_values: dict[str, set[str]] = {name: set() for name in categorical_features}
    batch_count = 0
    for batch in batches:
        batch_count += 1
        _require_columns(batch, (*numeric_features, *categorical_features))
        for name in numeric_features:
            for value in batch.column(name).to_pylist():
                if value is not None:
                    numeric_statistics[name].add(float(value), feature_name=name)
        for name in categorical_features:
            categorical_values[name].update(
                str(value) for value in batch.column(name).to_pylist() if value is not None
            )
            if len(categorical_values[name]) > max_vocabulary_size:
                raise ValueError(f"categorical vocabulary exceeds limit: {name}")
    if batch_count == 0:
        raise ValueError("cannot fit feature state from empty batches")

    numeric_state: list[NumericFeatureState] = []
    for name in numeric_features:
        statistics = numeric_statistics[name]
        if statistics.count == 0:
            raise ValueError(f"numeric feature has no valid values: {name}")
        variance = statistics.squared_deviation_sum / statistics.count
        numeric_state.append(
            NumericFeatureState(
                name=name,
                mean=statistics.mean,
                stddev=math.sqrt(variance) if variance > 0.0 else 1.0,
                valid_count=statistics.count,
            )
        )
    categorical_state = tuple(
        CategoricalFeatureState(name=name, vocabulary=tuple(sorted(categorical_values[name])))
        for name in categorical_features
    )
    return FittedFeatureState(
        feature_schema_version=feature_schema_version,
        numeric=tuple(numeric_state),
        categorical=categorical_state,
    )


def transform_batch(batch: pa.RecordBatch, state: FittedFeatureState) -> RerankBatch:
    feature_names = tuple(item.name for item in state.numeric) + tuple(
        item.name for item in state.categorical
    )
    label_columns = (
        "effective_watch",
        "effective_watch_valid",
        "non_fast_swipe",
        "non_fast_swipe_valid",
        "immersive_click",
        "immersive_click_valid",
        "watch_duration_ms",
        "content_duration_ms",
    )
    _require_columns(batch, (*feature_names, *label_columns))

    normalized_columns: list[list[float]] = []
    missing_columns: list[list[float]] = []
    for numeric in state.numeric:
        normalized: list[float] = []
        missing: list[float] = []
        for value in batch.column(numeric.name).to_pylist():
            if value is None:
                normalized.append(0.0)
                missing.append(1.0)
            else:
                normalized.append((float(value) - numeric.mean) / numeric.stddev)
                missing.append(0.0)
        normalized_columns.append(normalized)
        missing_columns.append(missing)
    numeric_rows = list(zip(*normalized_columns, *missing_columns, strict=True))

    categorical_columns: list[list[int]] = []
    for categorical in state.categorical:
        lookup = {value: index + 2 for index, value in enumerate(categorical.vocabulary)}
        categorical_columns.append(
            [
                categorical.missing_index
                if value is None
                else lookup.get(str(value), categorical.oov_index)
                for value in batch.column(categorical.name).to_pylist()
            ]
        )
    categorical_rows = (
        list(zip(*categorical_columns, strict=True))
        if categorical_columns
        else [tuple() for _ in range(batch.num_rows)]
    )

    labels: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for target in TARGETS:
        if target == "completion":
            values, valid = completion_labels(
                batch.column("watch_duration_ms").to_numpy(zero_copy_only=False),
                batch.column("content_duration_ms").to_numpy(zero_copy_only=False),
            )
            labels[target] = torch.tensor(values, dtype=torch.float32)
            masks[target] = torch.tensor(valid, dtype=torch.bool)
        else:
            labels[target] = torch.tensor(batch.column(target).to_pylist(), dtype=torch.float32)
            masks[target] = torch.tensor(
                batch.column(f"{target}_valid").to_pylist(), dtype=torch.bool
            )
    return RerankBatch(
        numeric=torch.tensor(numeric_rows, dtype=torch.float32),
        categorical=torch.tensor(categorical_rows, dtype=torch.long),
        labels=labels,
        masks=masks,
    )
