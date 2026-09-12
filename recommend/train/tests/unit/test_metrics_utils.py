from __future__ import annotations

import pytest
import torch

from recommend.train.comm.metrics_utils import MetricStatus, compute_target_metrics


def test_target_metrics_report_counts_loss_and_auc() -> None:
    metrics = compute_target_metrics(
        "completion",
        logits=torch.tensor([-2.0, 2.0, -1.0, 1.0]),
        labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        mask=torch.tensor([True, True, True, True]),
        minimum_valid_rows=4,
    )
    assert metrics.status is MetricStatus.OK
    assert metrics.valid_count == 4
    assert metrics.positive_count == 2
    assert metrics.negative_count == 2
    assert metrics.loss is not None and metrics.loss > 0.0
    assert metrics.auc == 1.0
    assert metrics.metric_version == "binary-v1"


@pytest.mark.parametrize(
    ("labels", "minimum"),
    [([1.0, 1.0], 2), ([0.0, 1.0], 3)],
)
def test_target_metrics_mark_single_class_or_small_sample_not_evaluable(
    labels: list[float], minimum: int
) -> None:
    metrics = compute_target_metrics(
        "completion",
        logits=torch.zeros(2),
        labels=torch.tensor(labels),
        mask=torch.tensor([True, True]),
        minimum_valid_rows=minimum,
    )
    assert metrics.status is MetricStatus.NOT_EVALUABLE
    assert metrics.auc is None
    assert metrics.loss is not None
