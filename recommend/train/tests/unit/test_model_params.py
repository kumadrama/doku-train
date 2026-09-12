from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recommend.train.comm.model_params import TrainConfig
from recommend.train.comm.model_params_validator import schema_matches, validate_config_payload
from recommend.train.models.rerank.model_params import RerankModelParams


def valid_payload() -> dict[str, object]:
    return {
        "model_name": "rerank",
        "checkpoint_dir": "runs/test",
        "seed": 7,
        "batch_size": 128,
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "bad_row_policy": "fail",
        "execution": {
            "strategy": "single_process_cpu",
            "intra_op_threads": 8,
            "inter_op_threads": 1,
            "reader_workers": 0,
            "prefetch_batches": 1,
            "memory_limit_bytes": 8_589_934_592,
        },
        "split": {"train_days": 27, "validation_days": 1, "refit_days": 28},
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001},
        "metrics": {"minimum_valid_rows": 10, "auc_warning_floor": 0.5},
        "model_params": {
            "numeric_features": ["age_hours", "watch_7d"],
            "categorical_features": ["country"],
            "categorical_cardinalities": [8],
            "embedding_dim": 4,
            "shared_hidden_dims": [32, 16],
            "tower_hidden_dim": 8,
            "activation": "relu",
            "dropout": 0.1,
            "loss_weights": {
                "effective_watch": 1.0,
                "completion": 1.0,
                "non_fast_swipe": 1.0,
                "immersive_click": 1.0,
            },
        },
    }


def test_unknown_common_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_config_payload(valid_payload() | {"cuda": True}, RerankModelParams)


def test_only_single_process_cpu_is_accepted() -> None:
    payload = valid_payload()
    payload["execution"] = {**payload["execution"], "strategy": "ddp"}  # type: ignore[misc]
    with pytest.raises(ValidationError, match="single_process_cpu"):
        validate_config_payload(payload, RerankModelParams)


def test_invalid_resource_budget_is_rejected() -> None:
    payload = valid_payload()
    payload["execution"] = {**payload["execution"], "memory_limit_bytes": 0}  # type: ignore[misc]
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        validate_config_payload(payload, RerankModelParams)


def test_loss_weights_require_exactly_four_targets() -> None:
    payload = valid_payload()
    model_params = dict(payload["model_params"])  # type: ignore[arg-type]
    model_params["loss_weights"] = {"completion": 1.0}
    payload["model_params"] = model_params
    with pytest.raises(ValidationError, match="four positive target weights"):
        validate_config_payload(payload, RerankModelParams)


def test_valid_payload_has_no_export_policy() -> None:
    config = validate_config_payload(valid_payload(), RerankModelParams)
    assert config.common.split.refit_days == 28
    assert config.model.shared_hidden_dims == (32, 16)
    assert "export" not in TrainConfig.model_fields


def test_checked_schema_snapshots_match_models() -> None:
    assert schema_matches(TrainConfig, Path("recommend/train/comm/config.schema.yml"))
    assert schema_matches(
        RerankModelParams, Path("recommend/train/models/rerank/config.schema.yml")
    )
