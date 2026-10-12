"""Task metrics report facts; no quality gates or implicit thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from recommend.train.comm.runtime_budget import check_memory


def metric_values(
    task: dict[str, Any], name: str, arrays: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    check_memory(prediction.size * 64)
    result: dict[str, Any] = {}
    for key, m in task["metrics"].items():
        mask = arrays["valid:" + name + ":" + key][:, 0].astype(bool)
        y = arrays.get("value:" + name + ":" + key, arrays["metric:" + name])[:, 0][mask]
        p = prediction[mask]
        n = len(y)
        func = m["func"]
        value: float | int | None = None
        reason = None
        if not n:
            reason = "no_valid_samples"
        elif not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("non-finite evaluation value")
        elif func == "valid_rows":
            value = n
        elif func in {"positive_rate", "mean"}:
            value = float(y.mean())
        elif func == "auc":
            if len(np.unique(y)) < 2:
                reason = "single_class"
            else:
                value = float(roc_auc_score(y, p))
        elif func == "logloss":
            if not np.isin(y, [0, 1]).all():
                raise ValueError("logloss requires binary labels")
            q = np.clip(p.astype(np.float64), 1e-15, 1 - 1e-15)
            value = float(-(y * np.log(q) + (1 - y) * np.log1p(-q)).mean())
        elif func == "mae":
            value = float(np.abs(y - p).mean())
        elif func == "mse":
            value = float(np.square(y.astype(np.float64) - p).mean())
        elif func == "rmse":
            value = float(np.sqrt(np.square(y.astype(np.float64) - p).mean()))
        else:
            raise ValueError("unimplemented metric")
        result[key] = {"value": value, "reason": reason, "valid_rows": n}
    result["label_rows"] = int(arrays["mask:" + name].sum())
    return result
