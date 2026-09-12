from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base configuration that rejects misspelled and unsupported fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(StrictModel):
    strategy: Literal["single_process_cpu"]
    intra_op_threads: int = Field(ge=1)
    inter_op_threads: int = Field(ge=1)
    reader_workers: Literal[0]
    prefetch_batches: Literal[1]
    memory_limit_bytes: int = Field(ge=1)


class SplitConfig(StrictModel):
    train_days: Literal[27] = 27
    validation_days: Literal[1] = 1
    refit_days: Literal[28] = 28


class OptimizerConfig(StrictModel):
    name: Literal["adamw"] = "adamw"
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class MetricsConfig(StrictModel):
    minimum_valid_rows: int = Field(ge=2)
    auc_warning_floor: float = Field(ge=0.0, le=1.0)


class TrainConfig(StrictModel):
    model_name: str = Field(min_length=1)
    checkpoint_dir: Path
    seed: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=1)
    bad_row_policy: Literal["fail"]
    execution: ExecutionConfig
    split: SplitConfig
    optimizer: OptimizerConfig
    metrics: MetricsConfig
    model_params: dict[str, object]
