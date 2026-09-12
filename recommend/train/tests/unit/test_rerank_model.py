from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS, RerankModelParams
from recommend.train.models.rerank.rerank_model import RerankModel, masked_multitask_loss


def params() -> RerankModelParams:
    return RerankModelParams(
        numeric_features=("age_hours", "watch_7d"),
        categorical_features=("country",),
        categorical_cardinalities=(4,),
        embedding_dim=3,
        shared_hidden_dims=(8, 4),
        tower_hidden_dim=2,
        activation="relu",
        dropout=0.0,
        loss_weights={target: 1.0 for target in TARGETS},
    )


def batch() -> RerankBatch:
    return RerankBatch(
        numeric=torch.zeros(3, 4),
        categorical=torch.tensor([[0], [1], [2]]),
        labels={target: torch.tensor([0.0, 1.0, 0.0]) for target in TARGETS},
        masks={target: torch.tensor([True, True, True]) for target in TARGETS},
    )


def test_shared_bottom_model_returns_four_stable_logits() -> None:
    model = RerankModel(params())
    output = model(batch().numeric, batch().categorical)
    assert tuple(output) == TARGETS
    assert all(value.shape == (3,) for value in output.values())


def test_masked_loss_ignores_invalid_targets_and_rows() -> None:
    logits = {target: torch.tensor([0.0, 1.0, -1.0]) for target in TARGETS}
    labels = {target: torch.tensor([0.0, 1.0, 1.0]) for target in TARGETS}
    masks = {target: torch.tensor([False, False, False]) for target in TARGETS}
    masks["completion"] = torch.tensor([True, False, True])
    actual = masked_multitask_loss(
        logits,
        labels,
        masks,
        {target: 1.0 for target in TARGETS},
    )
    expected = F.binary_cross_entropy_with_logits(
        logits["completion"][[0, 2]], labels["completion"][[0, 2]]
    )
    assert torch.equal(actual, expected)


def test_masked_loss_rejects_batch_without_active_target() -> None:
    logits = {target: torch.zeros(2) for target in TARGETS}
    labels = {target: torch.zeros(2) for target in TARGETS}
    masks = {target: torch.zeros(2, dtype=torch.bool) for target in TARGETS}
    with pytest.raises(ValueError, match="no active target"):
        masked_multitask_loss(logits, labels, masks, params().loss_weights)


def test_model_module_has_no_storage_or_process_side_effects() -> None:
    source = Path("recommend/train/models/rerank/rerank_model.py").read_text()
    for forbidden in ("boto3", "subprocess", "os.environ", "datasvr", "train.run"):
        assert forbidden not in source
