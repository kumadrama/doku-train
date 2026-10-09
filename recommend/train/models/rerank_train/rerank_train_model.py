from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass

import psutil

# XGBoost must initialize OpenMP before PyTorch on macOS.
# isort: off
import xgboost as xgb

# isort: on
import torch

from recommend.train.comm.errors import PipelineBlocked
from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS
from recommend.train.models.rerank_train.data_utils import (
    BatchFactory,
    build_quantile_dmatrix,
    feature_names,
    feature_types,
    prediction_dmatrix,
)
from recommend.train.models.rerank_train.model_params import RerankTrainModelParams

ARTIFACT_FORMAT_VERSION = "1"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TargetTrainingTrace:
    target: str
    selected_rounds: int
    validation_logloss: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EvaluationFitResult:
    model: RerankTrainModel
    selected_rounds: dict[str, int]
    traces: tuple[TargetTrainingTrace, ...]


class RerankTrainModel:
    def __init__(
        self,
        params: RerankTrainModelParams,
        boosters: Mapping[str, xgb.Booster],
        rounds: Mapping[str, int],
    ) -> None:
        if set(boosters) != set(TARGETS) or set(rounds) != set(TARGETS):
            raise ValueError("XGBoost model requires exactly four targets")
        if any(int(rounds[target]) < 1 for target in TARGETS):
            raise ValueError("XGBoost target rounds must be positive")
        self.params = params
        self.boosters = {target: boosters[target] for target in TARGETS}
        self.rounds = {target: int(rounds[target]) for target in TARGETS}

    def logits(self, batch: RerankBatch) -> dict[str, torch.Tensor]:
        matrix = prediction_dmatrix(batch, self.params)
        logits: dict[str, torch.Tensor] = {}
        for target in TARGETS:
            values = self.boosters[target].predict(
                matrix,
                output_margin=True,
                iteration_range=(0, self.rounds[target]),
                validate_features=True,
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"NUMERICAL_FAILURE: non-finite prediction for {target}")
            logits[target] = torch.from_numpy(values.copy()).to(dtype=torch.float32)
        return logits

    def to_bytes(self) -> bytes:
        manifest = json.dumps(
            {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "model_name": "rerank_train",
                "targets": list(TARGETS),
                "rounds": self.rounds,
                "feature_names": list(feature_names(self.params)),
                "feature_types": list(feature_types(self.params)),
                "xgboost_version": xgb.__version__,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w") as archive:
            self._write_entry(archive, "manifest.json", manifest)
            for target in TARGETS:
                self._write_entry(
                    archive,
                    f"models/{target}.ubj",
                    bytes(self.boosters[target].save_raw(raw_format="ubj")),
                )
        return output.getvalue()

    @classmethod
    def from_bytes(
        cls,
        body: bytes,
        params: RerankTrainModelParams,
    ) -> RerankTrainModel:
        if not body or len(body) > MAX_ARTIFACT_BYTES:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: invalid XGBoost artifact size")
        expected_names = {"manifest.json", *(f"models/{target}.ubj" for target in TARGETS)}
        try:
            with zipfile.ZipFile(io.BytesIO(body), mode="r") as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != expected_names:
                    raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost artifact entries")
                if any(name.startswith("/") or ".." in name.split("/") for name in names):
                    raise ValueError("CHECKPOINT_INCOMPATIBLE: unsafe XGBoost artifact entry")
                manifest = json.loads(archive.read("manifest.json"))
                if not isinstance(manifest, dict):
                    raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost manifest")
                cls._validate_manifest(manifest, params)
                boosters: dict[str, xgb.Booster] = {}
                for target in TARGETS:
                    booster = xgb.Booster()
                    booster.load_model(bytearray(archive.read(f"models/{target}.ubj")))
                    boosters[target] = booster
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: unreadable XGBoost artifact") from error
        rounds_payload = manifest["rounds"]
        if not isinstance(rounds_payload, dict):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost rounds")
        rounds = {target: int(rounds_payload[target]) for target in TARGETS}
        return cls(params, boosters, rounds)

    @staticmethod
    def _write_entry(archive: zipfile.ZipFile, name: str, body: bytes) -> None:
        entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_STORED
        entry.external_attr = 0o600 << 16
        archive.writestr(entry, body)

    @staticmethod
    def _validate_manifest(manifest: dict[str, object], params: RerankTrainModelParams) -> None:
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost format version")
        if manifest.get("model_name") != "rerank_train":
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost model name")
        if manifest.get("targets") != list(TARGETS):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost targets")
        if manifest.get("feature_names") != list(feature_names(params)):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost feature names")
        if manifest.get("feature_types") != list(feature_types(params)):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost feature types")
        rounds = manifest.get("rounds")
        if not isinstance(rounds, dict) or set(rounds) != set(TARGETS):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost rounds")
        if any(
            isinstance(rounds[target], bool)
            or not isinstance(rounds[target], int)
            or rounds[target] < 1
            for target in TARGETS
        ):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: XGBoost rounds")


class RerankTrainTrainer:
    def __init__(
        self,
        params: RerankTrainModelParams,
        *,
        seed: int,
        nthread: int,
        memory_limit_bytes: int,
    ) -> None:
        self._params = params
        self._seed = seed
        self._nthread = nthread
        self._memory_limit_bytes = memory_limit_bytes

    def fit_evaluation(
        self,
        train_batches: BatchFactory,
        validation_batches: BatchFactory,
    ) -> EvaluationFitResult:
        boosters: dict[str, xgb.Booster] = {}
        selected_rounds: dict[str, int] = {}
        traces: list[TargetTrainingTrace] = []
        for target_index, target in enumerate(TARGETS):
            self._check_memory("evaluate")
            train = build_quantile_dmatrix(
                train_batches,
                target,
                self._params,
                nthread=self._nthread,
            )
            validation = build_quantile_dmatrix(
                validation_batches,
                target,
                self._params,
                nthread=self._nthread,
                ref=train,
            )
            self._check_memory("evaluate")
            history: dict[str, dict[str, list[float]]] = {}
            booster = xgb.train(
                self._booster_params(self._seed + target_index),
                train,
                num_boost_round=self._params.num_boost_round,
                evals=[(validation, "validation")],
                early_stopping_rounds=self._params.early_stopping_rounds,
                evals_result=history,
                verbose_eval=False,
            )
            rounds = int(booster.best_iteration) + 1
            boosters[target] = booster
            selected_rounds[target] = rounds
            traces.append(
                TargetTrainingTrace(
                    target=target,
                    selected_rounds=rounds,
                    validation_logloss=tuple(history["validation"]["logloss"]),
                )
            )
            self._check_memory("evaluate")
        return EvaluationFitResult(
            model=RerankTrainModel(self._params, boosters, selected_rounds),
            selected_rounds=selected_rounds,
            traces=tuple(traces),
        )

    def fit_production(
        self,
        refit_batches: BatchFactory,
        selected_rounds: Mapping[str, int],
    ) -> RerankTrainModel:
        if set(selected_rounds) != set(TARGETS):
            raise ValueError("production refit requires four selected round counts")
        boosters: dict[str, xgb.Booster] = {}
        for target_index, target in enumerate(TARGETS):
            self._check_memory("refit")
            refit = build_quantile_dmatrix(
                refit_batches,
                target,
                self._params,
                nthread=self._nthread,
            )
            self._check_memory("refit")
            boosters[target] = xgb.train(
                self._booster_params(self._seed + 10_000 + target_index),
                refit,
                num_boost_round=int(selected_rounds[target]),
                verbose_eval=False,
            )
            self._check_memory("refit")
        return RerankTrainModel(self._params, boosters, selected_rounds)

    def _booster_params(self, seed: int) -> dict[str, object]:
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": self._nthread,
            "seed": seed,
            "learning_rate": self._params.learning_rate,
            "max_depth": self._params.max_depth,
            "min_child_weight": self._params.min_child_weight,
            "subsample": self._params.subsample,
            "colsample_bytree": self._params.colsample_bytree,
            "reg_alpha": self._params.reg_alpha,
            "reg_lambda": self._params.reg_lambda,
            "max_bin": self._params.max_bin,
            "max_cat_to_onehot": self._params.max_cat_to_onehot,
            "validate_parameters": True,
        }

    def _check_memory(self, stage: str) -> None:
        if psutil.Process().memory_info().rss > self._memory_limit_bytes:
            raise PipelineBlocked("RESOURCE_BUDGET_EXCEEDED", stage=stage)
