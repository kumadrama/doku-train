from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

from recommend.train.comm.metrics_utils import MetricSet, compute_target_metrics
from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS


class RerankLogitAdapter(Protocol):
    def logits(self, batch: RerankBatch) -> dict[str, torch.Tensor]: ...


def evaluate_rerank(
    adapter: RerankLogitAdapter,
    batches: Iterable[RerankBatch],
    *,
    minimum_valid_rows: int,
    validation_slice_digest: str,
) -> MetricSet:
    logits: dict[str, list[torch.Tensor]] = {target: [] for target in TARGETS}
    labels: dict[str, list[torch.Tensor]] = {target: [] for target in TARGETS}
    masks: dict[str, list[torch.Tensor]] = {target: [] for target in TARGETS}
    with torch.inference_mode():
        for batch in batches:
            output = adapter.logits(batch)
            for target in TARGETS:
                logits[target].append(output[target].detach().cpu())
                labels[target].append(batch.labels[target].detach().cpu())
                masks[target].append(batch.masks[target].detach().cpu())
    if any(not values for values in logits.values()):
        raise ValueError("DATA_EXHAUSTED: validation produced no batches")
    metrics = tuple(
        compute_target_metrics(
            target,
            logits=torch.cat(logits[target]),
            labels=torch.cat(labels[target]),
            mask=torch.cat(masks[target]),
            minimum_valid_rows=minimum_valid_rows,
        )
        for target in TARGETS
    )
    return MetricSet(validation_slice_digest=validation_slice_digest, targets=metrics)
