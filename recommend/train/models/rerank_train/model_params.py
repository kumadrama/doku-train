from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from recommend.train.comm.model_params import (
    ExecutionConfig,
    MetricsConfig,
    SplitConfig,
    StrictModel,
)


class RerankTrainModelParams(StrictModel):
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    num_boost_round: int = Field(ge=1, le=10_000)
    early_stopping_rounds: int = Field(ge=1, le=1_000)
    learning_rate: float = Field(gt=0.0, le=1.0)
    max_depth: int = Field(ge=1, le=32)
    min_child_weight: float = Field(ge=0.0)
    subsample: float = Field(gt=0.0, le=1.0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    reg_alpha: float = Field(ge=0.0)
    reg_lambda: float = Field(ge=0.0)
    max_bin: int = Field(ge=2, le=65_536)
    max_cat_to_onehot: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_features(self) -> RerankTrainModelParams:
        features = self.numeric_features + self.categorical_features
        if not features:
            raise ValueError("at least one model feature is required")
        if len(features) != len(set(features)):
            raise ValueError("feature names must be unique")
        if self.early_stopping_rounds > self.num_boost_round:
            raise ValueError("early_stopping_rounds must not exceed num_boost_round")
        return self


class RerankTrainConfig(StrictModel):
    model_name: Literal["rerank_train"]
    checkpoint_dir: Path
    seed: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    bad_row_policy: Literal["fail"]
    execution: ExecutionConfig
    split: SplitConfig
    metrics: MetricsConfig
    model_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedRerankTrainConfig:
    common: RerankTrainConfig
    model: RerankTrainModelParams


def load_rerank_train_config(path: Path) -> ValidatedRerankTrainConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    common = RerankTrainConfig.model_validate(payload)
    model = RerankTrainModelParams.model_validate(common.model_params)
    return ValidatedRerankTrainConfig(common=common, model=model)
