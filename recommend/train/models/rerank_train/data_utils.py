from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS
from recommend.train.models.rerank_train.model_params import RerankTrainModelParams

BatchFactory = Callable[[], Iterable[RerankBatch]]


def feature_names(params: RerankTrainModelParams) -> tuple[str, ...]:
    return (
        *params.numeric_features,
        *(f"{name}__missing" for name in params.numeric_features),
        *params.categorical_features,
    )


def feature_types(params: RerankTrainModelParams) -> tuple[str, ...]:
    return (
        *("q" for _ in params.numeric_features),
        *("q" for _ in params.numeric_features),
        *("c" for _ in params.categorical_features),
    )


def batch_matrix(batch: RerankBatch) -> npt.NDArray[np.float32]:
    numeric = batch.numeric.detach().cpu().numpy().astype(np.float32, copy=False)
    categorical = batch.categorical.detach().cpu().numpy().astype(np.float32, copy=False)
    matrix = np.concatenate((numeric, categorical), axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("NUMERICAL_FAILURE: non-finite XGBoost feature")
    return matrix


class RerankDataIter(xgb.DataIter):
    def __init__(
        self,
        batch_factory: BatchFactory,
        target: str,
        params: RerankTrainModelParams,
    ) -> None:
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}")
        self._batch_factory = batch_factory
        self._target = target
        self._feature_names = feature_names(params)
        self._feature_types = feature_types(params)
        self._iterator: Iterator[RerankBatch] = iter(())
        super().__init__(release_data=True)
        self.reset()

    def reset(self) -> None:
        self._iterator = iter(self._batch_factory())

    def next(self, input_data: Any) -> bool:
        for batch in self._iterator:
            mask = batch.masks[self._target].detach().cpu().numpy().astype(np.bool_, copy=False)
            if not mask.any():
                continue
            labels = (
                batch.labels[self._target]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)[mask]
            )
            if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
                raise ValueError(f"invalid binary labels for {self._target}")
            input_data(
                data=batch_matrix(batch)[mask],
                label=labels,
                feature_names=self._feature_names,
                feature_types=self._feature_types,
            )
            return True
        return False


def build_quantile_dmatrix(
    batch_factory: BatchFactory,
    target: str,
    params: RerankTrainModelParams,
    *,
    nthread: int,
    ref: xgb.QuantileDMatrix | None = None,
) -> xgb.QuantileDMatrix:
    matrix = xgb.QuantileDMatrix(
        RerankDataIter(batch_factory, target, params),
        nthread=nthread,
        max_bin=params.max_bin,
        ref=ref,
        enable_categorical=True,
    )
    if matrix.num_row() == 0:
        raise ValueError(f"DATA_EXHAUSTED: no valid rows for {target}")
    return matrix


def prediction_dmatrix(batch: RerankBatch, params: RerankTrainModelParams) -> xgb.DMatrix:
    return xgb.DMatrix(
        batch_matrix(batch),
        feature_names=feature_names(params),
        feature_types=feature_types(params),
        enable_categorical=True,
    )
