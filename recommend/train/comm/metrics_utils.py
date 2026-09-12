from __future__ import annotations

from enum import StrEnum

import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from recommend.train.comm.model_params import StrictModel


class MetricStatus(StrEnum):
    OK = "OK"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class TargetMetrics(StrictModel):
    target: str
    metric_version: str
    status: MetricStatus
    valid_count: int
    positive_count: int
    negative_count: int
    loss: float | None
    auc: float | None


class MetricSet(StrictModel):
    validation_slice_digest: str
    targets: tuple[TargetMetrics, ...]

    def by_target(self) -> dict[str, TargetMetrics]:
        return {metric.target: metric for metric in self.targets}


def compute_target_metrics(
    target: str,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    minimum_valid_rows: int,
) -> TargetMetrics:
    if logits.shape != labels.shape or mask.shape != labels.shape:
        raise ValueError(f"metric tensor shape mismatch: {target}")
    selected_logits = logits.detach().cpu()[mask.detach().cpu().bool()]
    selected_labels = labels.detach().cpu()[mask.detach().cpu().bool()]
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_labels).all():
        raise ValueError(f"NUMERICAL_FAILURE: non-finite metric input for {target}")
    valid_count = int(selected_labels.numel())
    positive_count = int((selected_labels == 1.0).sum().item())
    negative_count = int((selected_labels == 0.0).sum().item())
    if positive_count + negative_count != valid_count:
        raise ValueError(f"metric labels must be binary: {target}")
    loss = (
        float(F.binary_cross_entropy_with_logits(selected_logits, selected_labels).item())
        if valid_count
        else None
    )
    evaluable = valid_count >= minimum_valid_rows and positive_count > 0 and negative_count > 0
    auc = None
    if evaluable:
        auc = float(
            roc_auc_score(
                selected_labels.numpy(),
                torch.sigmoid(selected_logits).numpy(),
            )
        )
    return TargetMetrics(
        target=target,
        metric_version="binary-v1",
        status=MetricStatus.OK if evaluable else MetricStatus.NOT_EVALUABLE,
        valid_count=valid_count,
        positive_count=positive_count,
        negative_count=negative_count,
        loss=loss,
        auc=auc,
    )
