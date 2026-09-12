from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import torch

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, ShardManifest


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: tuple[ShardManifest, ...]
    validation: tuple[ShardManifest, ...]
    refit: tuple[ShardManifest, ...]


@dataclass(frozen=True, slots=True)
class RerankBatch:
    numeric: torch.Tensor
    categorical: torch.Tensor
    labels: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


def build_daily_split(manifest: DatasetManifest) -> TemporalSplit:
    shards = tuple(sorted(manifest.shards, key=lambda item: (item.event_date, item.uri)))
    event_dates = tuple(sorted({shard.event_date for shard in shards}))
    if len(event_dates) != 28:
        raise ValueError("SPLIT_INVALID: daily training requires 28 distinct days")
    shards_by_date = {
        event_date: tuple(shard for shard in shards if shard.event_date == event_date)
        for event_date in event_dates
    }
    for previous, current in pairwise(event_dates):
        if current != previous + timedelta(days=1):
            raise ValueError("SPLIT_INVALID: event dates must be consecutive")
        previous_max = max(shard.max_event_time_ms for shard in shards_by_date[previous])
        current_min = min(shard.min_event_time_ms for shard in shards_by_date[current])
        if previous_max >= current_min:
            raise ValueError("SPLIT_INVALID: shard event ranges overlap")
    maturity_cutoff_ms = manifest.as_of_ms - manifest.label_maturity_hours * 3_600_000
    if any(shard.max_event_time_ms > maturity_cutoff_ms for shard in shards):
        raise ValueError("LABEL_NOT_MATURE: shard is newer than the maturity cutoff")
    training_dates = frozenset(event_dates[:27])
    train = tuple(shard for shard in shards if shard.event_date in training_dates)
    validation = tuple(shard for shard in shards if shard.event_date == event_dates[-1])
    return TemporalSplit(train=train, validation=validation, refit=shards)


def completion_labels(
    watch_duration_ms: npt.ArrayLike,
    content_duration_ms: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    watch = np.asarray(watch_duration_ms, dtype=np.float64)
    content = np.asarray(content_duration_ms, dtype=np.float64)
    if watch.shape != content.shape:
        raise ValueError("watch and content durations must have equal shapes")
    valid = np.isfinite(watch) & np.isfinite(content) & (watch >= 0.0) & (content > 0.0)
    labels = np.zeros(watch.shape, dtype=np.float32)
    labels[valid] = (watch[valid] / content[valid] >= 0.95).astype(np.float32)
    return labels, valid
