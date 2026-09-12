from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from recommend.train.comm.model_params import StrictModel

TARGETS = ("effective_watch", "completion", "non_fast_swipe", "immersive_click")


class RerankModelParams(StrictModel):
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    embedding_dim: int = Field(ge=1, le=256)
    shared_hidden_dims: tuple[int, ...]
    tower_hidden_dim: int = Field(ge=1, le=4096)
    activation: Literal["relu", "silu"]
    dropout: float = Field(ge=0.0, lt=1.0)
    loss_weights: dict[str, float]

    @model_validator(mode="after")
    def validate_model_shape(self) -> RerankModelParams:
        if not self.numeric_features:
            raise ValueError("numeric_features must not be empty")
        if len(set(self.numeric_features + self.categorical_features)) != len(
            self.numeric_features + self.categorical_features
        ):
            raise ValueError("feature names must be unique")
        if len(self.categorical_features) != len(self.categorical_cardinalities):
            raise ValueError("categorical features and cardinalities must have equal length")
        if any(size < 2 for size in self.categorical_cardinalities):
            raise ValueError("categorical cardinality must reserve missing and OOV")
        if (
            not self.shared_hidden_dims
            or len(self.shared_hidden_dims) > 5
            or any(size < 1 or size > 4096 for size in self.shared_hidden_dims)
        ):
            raise ValueError("shared_hidden_dims must contain 1-5 values in [1, 4096]")
        if set(self.loss_weights) != set(TARGETS) or any(
            weight <= 0.0 for weight in self.loss_weights.values()
        ):
            raise ValueError("loss_weights must contain four positive target weights")
        return self
