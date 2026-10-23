"""Public neural loss entry point, following Finder comm.nn_utils.get_loss."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.nn import functional as F


def get_loss(
    head: dict[str, Any],
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_weight_data: torch.Tensor | float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Dispatch task loss and reduce weighted values through one shared entry.

    Callers resolve configured expressions and masks before entering this function.
    Evaluation passes unit weights, preserving its unweighted reporting contract.
    """
    values = _loss_values(logits, labels, head) * loss_weight_data
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean() if values.numel() else values.sum()
    raise ValueError("unsupported loss reduction")


def _loss_values(logits: torch.Tensor, label: torch.Tensor, task: dict[str, Any]) -> torch.Tensor:
    if task["loss_type"] == "logloss":
        if not ((label == 0) | (label == 1)).all():
            raise ValueError("binary labels must be zero or one")
        return F.binary_cross_entropy_with_logits(logits[:, 0], label, reduction="none")
    if task["loss_type"] == "mseloss":
        return (logits[:, 0] - label).square()
    if task["loss_type"] != "softmax_soft":
        raise ValueError("unsupported loss_type")
    soft = task["soft_label"]
    bins = torch.arange(task["num_outputs"], device=label.device, dtype=label.dtype) * soft["scale"]
    logp = -0.5 * ((bins[None, :] - label[:, None]) / soft["sigma"]).square()
    target = torch.softmax(logp, dim=1)
    return -(target * F.log_softmax(logits, dim=1)).sum(1)
