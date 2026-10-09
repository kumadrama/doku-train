from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS
from recommend.train.models.rerank_train.data_utils import build_quantile_dmatrix
from recommend.train.models.rerank_train.model_params import RerankTrainModelParams
from recommend.train.models.rerank_train.rerank_train_model import (
    RerankTrainModel,
    RerankTrainTrainer,
)


def params() -> RerankTrainModelParams:
    return RerankTrainModelParams(
        numeric_features=("watch_7d",),
        categorical_features=("country",),
        num_boost_round=4,
        early_stopping_rounds=2,
        learning_rate=0.3,
        max_depth=2,
        min_child_weight=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_bin=16,
        max_cat_to_onehot=4,
    )


def batches() -> Iterable[RerankBatch]:
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    yield RerankBatch(
        numeric=torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [-0.8, 0.0],
                [0.8, 0.0],
                [-0.6, 0.0],
                [0.6, 0.0],
                [-0.4, 0.0],
                [0.4, 0.0],
            ]
        ),
        categorical=torch.tensor([[2], [3], [2], [3], [2], [3], [2], [3]]),
        labels={target: labels.clone() for target in TARGETS},
        masks={target: torch.ones(8, dtype=torch.bool) for target in TARGETS},
    )


def test_four_target_xgboost_round_trips_deterministic_checkpoint() -> None:
    trainer = RerankTrainTrainer(
        params(),
        seed=7,
        nthread=1,
        memory_limit_bytes=2_000_000_000,
    )
    evaluation = trainer.fit_evaluation(batches, batches)
    assert set(evaluation.selected_rounds) == set(TARGETS)
    assert all(rounds >= 1 for rounds in evaluation.selected_rounds.values())
    production = trainer.fit_production(batches, evaluation.selected_rounds)
    encoded = production.to_bytes()
    assert encoded == production.to_bytes()
    restored = RerankTrainModel.from_bytes(encoded, params())
    logits = restored.logits(next(iter(batches())))
    assert tuple(logits) == TARGETS
    assert all(values.shape == (8,) for values in logits.values())
    assert all(torch.isfinite(values).all() for values in logits.values())


def test_xgboost_checkpoint_rejects_unreadable_input() -> None:
    with pytest.raises(ValueError, match="CHECKPOINT_INCOMPATIBLE"):
        RerankTrainModel.from_bytes(b"not-a-zip", params())


def test_xgboost_checkpoint_rejects_unknown_entries() -> None:
    trainer = RerankTrainTrainer(
        params(),
        seed=7,
        nthread=1,
        memory_limit_bytes=2_000_000_000,
    )
    model = trainer.fit_production(
        batches,
        {target: 1 for target in TARGETS},
    )
    body = io.BytesIO(model.to_bytes())
    with zipfile.ZipFile(body, mode="a") as archive:
        archive.writestr("unexpected.txt", b"unexpected")
    with pytest.raises(ValueError, match="CHECKPOINT_INCOMPATIBLE"):
        RerankTrainModel.from_bytes(body.getvalue(), params())


def test_xgboost_checkpoint_rejects_missing_target() -> None:
    trainer = RerankTrainTrainer(
        params(),
        seed=7,
        nthread=1,
        memory_limit_bytes=2_000_000_000,
    )
    model = trainer.fit_production(
        batches,
        {target: 1 for target in TARGETS},
    )
    source = zipfile.ZipFile(io.BytesIO(model.to_bytes()), mode="r")
    body = io.BytesIO()
    with source, zipfile.ZipFile(body, mode="w") as destination:
        for name in source.namelist():
            if name != "models/immersive_click.ubj":
                destination.writestr(name, source.read(name))
    with pytest.raises(ValueError, match="CHECKPOINT_INCOMPATIBLE"):
        RerankTrainModel.from_bytes(body.getvalue(), params())


def test_xgboost_data_iter_applies_target_mask() -> None:
    original = next(iter(batches()))

    def masked_batches() -> Iterable[RerankBatch]:
        yield RerankBatch(
            numeric=original.numeric,
            categorical=original.categorical,
            labels=original.labels,
            masks={
                **original.masks,
                "effective_watch": torch.tensor(
                    [True, False, True, False, True, False, True, False]
                ),
            },
        )

    matrix = build_quantile_dmatrix(
        masked_batches,
        "effective_watch",
        params(),
        nthread=1,
    )
    assert matrix.num_row() == 4


def test_xgboost_params_reject_invalid_early_stopping() -> None:
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        RerankTrainModelParams.model_validate({**params().model_dump(), "early_stopping_rounds": 5})


def test_xgboost_model_plugin_has_no_storage_or_platform_side_effects() -> None:
    root = Path("recommend/train/models/rerank_train")
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    for forbidden in ("boto3", "sagemaker", "subprocess", "os.environ", "S3Storage"):
        assert forbidden not in source
