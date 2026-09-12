from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from recommend.train.models.rerank.model_params import TARGETS, RerankModelParams


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


class RerankModel(nn.Module):
    def __init__(self, params: RerankModelParams) -> None:
        super().__init__()
        self._categorical_count = len(params.categorical_cardinalities)
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, params.embedding_dim)
            for cardinality in params.categorical_cardinalities
        )
        input_dim = len(params.numeric_features) * 2 + (
            self._categorical_count * params.embedding_dim
        )
        shared_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in params.shared_hidden_dims:
            shared_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    _activation(params.activation),
                    nn.Dropout(params.dropout),
                ]
            )
            previous_dim = hidden_dim
        self.shared_bottom = nn.Sequential(*shared_layers)
        self.towers = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(previous_dim, params.tower_hidden_dim),
                    _activation(params.activation),
                    nn.Dropout(params.dropout),
                    nn.Linear(params.tower_hidden_dim, 1),
                )
                for target in TARGETS
            }
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> dict[str, torch.Tensor]:
        if numeric.ndim != 2 or categorical.ndim != 2:
            raise ValueError("numeric and categorical inputs must be rank two")
        if categorical.shape[1] != self._categorical_count:
            raise ValueError("categorical input width does not match model configuration")
        encoded = [numeric]
        encoded.extend(
            embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)
        )
        shared = self.shared_bottom(torch.cat(encoded, dim=1))
        return {target: self.towers[target](shared).squeeze(-1) for target in TARGETS}


def masked_multitask_loss(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> torch.Tensor:
    active_losses: list[torch.Tensor] = []
    active_weights: list[float] = []
    for target in TARGETS:
        if target not in logits or target not in labels or target not in masks:
            raise ValueError(f"missing target tensor: {target}")
        mask = masks[target].bool()
        if logits[target].shape != labels[target].shape or mask.shape != labels[target].shape:
            raise ValueError(f"target tensor shape mismatch: {target}")
        if mask.any():
            weight = float(weights[target])
            active_losses.append(
                F.binary_cross_entropy_with_logits(logits[target][mask], labels[target][mask])
                * weight
            )
            active_weights.append(weight)
    if not active_losses:
        raise ValueError("batch has no active target")
    return torch.stack(active_losses).sum() / sum(active_weights)
