from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Any

import numpy as np
import xgboost as xgb

from recommend.train.comm.checkpoint_agent import CheckpointAgent
from recommend.train.comm.runtime_budget import check_memory
from recommend.train.comm.task_metrics import metric_values


class BudgetCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Any) -> bool:
        check_memory()
        return False


class TreeRunner:
    checkpoint_name = "checkpoint.xgb.zip"

    def __init__(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        self.config = config
        self.state = state
        self.boosters: dict[str, xgb.Booster] = {}

    def matrix(self, arrays: dict[str, np.ndarray], name: str | None = None) -> xgb.DMatrix:
        check_memory(arrays["tree"].nbytes * 3)
        mask = arrays["mask:" + name][:, 0].astype(bool) if name else slice(None)
        types = ["q"] * len(self.config["tree_inputs"]["numeric"]) + ["c"] * len(
            self.config["tree_inputs"]["categorical"]
        )
        return xgb.DMatrix(
            arrays["tree"][mask],
            label=arrays["label:" + name][:, 0][mask] if name else None,
            weight=arrays["weight:" + name][:, 0][mask] if name else None,
            feature_types=types,
            enable_categorical=True,
            nthread=self.config["execution"]["intra_op_threads"],
        )

    def fit(
        self,
        train: dict[str, np.ndarray],
        validation: dict[str, np.ndarray],
        checkpoint: CheckpointAgent,
        identity: str,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if resume:
            raise ValueError(
                "tree resume requires a complete checkpoint; partial round resume is unsupported"
            )
        selected = {}
        history = {}
        for name, t in self.config["tasks"].items():
            if not train["mask:" + name].any():
                raise ValueError(f"no training labels: {name}")
            params = dict(self.config["model_params"])
            rounds = params.pop("num_boost_round")
            patience = params.pop("early_stopping_rounds")
            params.update(
                objective=t["objective"],
                device="cpu",
                nthread=self.config["execution"]["intra_op_threads"],
                seed=int.from_bytes(
                    hashlib.sha256(f"{self.config['seed']}:{name}".encode()).digest()[:4], "big"
                )
                % 2147483647,
                eval_metric=self.config["selection"]["per_task_metrics"][name],
            )
            if params["eval_metric"] == "mse":
                params["eval_metric"] = "rmse"
            data = self.matrix(train, name)
            evals = []
            if validation["mask:" + name].any():
                evals = [(self.matrix(validation, name), "validation")]
            log: dict[str, Any] = {}
            booster = xgb.train(
                params,
                data,
                num_boost_round=rounds,
                evals=evals,
                early_stopping_rounds=patience if evals else None,
                evals_result=log,
                verbose_eval=False,
                callbacks=[BudgetCallback()],
            )
            selected[name] = booster.best_iteration + 1 if evals else None
            self.boosters[name] = booster[: selected[name]] if selected[name] else booster
            history[name] = log
            print(f"task={name} selected_rounds={selected[name]}", flush=True)
        checkpoint.save_bytes(self.checkpoint_name, self.serialize(identity, selected))
        return {
            "selection": {
                "status": "selected" if all(selected.values()) else "unavailable",
                "rounds": selected,
            },
            "validation": self.evaluate(validation),
            "history": history,
        }

    def serialize(self, identity: str, selected: dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "format": "rerank-tree/v2",
                        "identity": identity,
                        "tasks": list(self.boosters),
                        "selected_rounds": selected,
                        "xgboost_version": xgb.__version__,
                    }
                ),
            )
            for i, booster in enumerate(self.boosters.values()):
                z.writestr(f"{i}.ubj", booster.save_raw(raw_format="ubj"))
        return buffer.getvalue()

    def evaluate(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        matrix = self.matrix(arrays)
        out = {}
        for name, t in self.config["tasks"].items():
            # Explicit margins avoid applying sigmoid twice to binary:logistic probabilities.
            margin = self.boosters[name].predict(matrix, output_margin=True)
            pred = np.exp(-np.logaddexp(0, -margin)) if t["predict_fn"] == "sigmoid" else margin
            metrics = metric_values(t, name, arrays, pred)
            metric = "logloss" if t["objective"] == "binary:logistic" else "mse"
            out[name] = {"metrics": metrics, "loss": metrics.get(metric, {}).get("value")}
        return out

    def load(self, payload: bytes) -> None:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            check_memory(sum(info.file_size for info in z.infolist()))
            metadata = json.loads(z.read("metadata.json"))
            if set(metadata["tasks"]) != set(self.config["tasks"]):
                raise ValueError("tree checkpoint tasks mismatch")
            expected = {"metadata.json"} | {f"{i}.ubj" for i in range(len(metadata["tasks"]))}
            if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
                raise ValueError("unexpected checkpoint entries")
            if metadata["xgboost_version"].split(".")[0] != xgb.__version__.split(".")[0]:
                raise ValueError("incompatible XGBoost checkpoint version")
            for i, name in enumerate(metadata["tasks"]):
                booster = xgb.Booster()
                booster.load_model(bytearray(z.read(f"{i}.ubj")))
                self.boosters[name] = booster
