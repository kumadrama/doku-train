from __future__ import annotations

from typing import Literal, Protocol

import torch

ModelRole = Literal["evaluation", "production"]


class ModelAdapter[BatchT](Protocol):
    module: torch.nn.Module

    def loss(self, batch: BatchT) -> torch.Tensor: ...

    def logits(self, batch: BatchT) -> dict[str, torch.Tensor]: ...


class AdapterFactory[BatchT](Protocol):
    def __call__(self, role: ModelRole) -> ModelAdapter[BatchT]: ...
