"""Four-target rerank model plugin."""

import torch

from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import RerankModelParams
from recommend.train.models.rerank.rerank_model import RerankModel, masked_multitask_loss


class RerankAdapter:
    def __init__(self, params: RerankModelParams) -> None:
        self.module = RerankModel(params)
        self._weights = params.loss_weights

    def loss(self, batch: RerankBatch) -> torch.Tensor:
        return masked_multitask_loss(self.logits(batch), batch.labels, batch.masks, self._weights)

    def logits(self, batch: RerankBatch) -> dict[str, torch.Tensor]:
        return self.module.forward(batch.numeric, batch.categorical)


__all__ = [
    "RerankAdapter",
    "RerankModel",
    "RerankModelParams",
    "masked_multitask_loss",
]
