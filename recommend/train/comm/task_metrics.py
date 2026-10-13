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
        elif func == "regression_auc":
            value = regression_auc(y, p)
            if value is None:
                reason = "no_comparable_pairs"
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


def regression_auc(y: np.ndarray, prediction: np.ndarray) -> float | None:
    """Exact unweighted concordance, O(n log n); equal labels are not compared."""
    order = np.argsort(y, kind="stable")
    labels = y[order]
    _, ranks = np.unique(prediction[order], return_inverse=True)
    tree = [0] * (int(ranks.max()) + 3) if len(ranks) else [0]

    def prefix(index: int) -> int:
        result = 0
        while index:
            result += tree[index]
            index -= index & -index
        return result

    score = 0
    pairs = 0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        for rank in ranks[start:end]:
            below = prefix(int(rank))
            equal = prefix(int(rank) + 1) - below
            score += 2 * below + equal
            pairs += start
        for rank in ranks[start:end]:
            index = int(rank) + 1
            while index < len(tree):
                tree[index] += 1
                index += index & -index
        start = end
    return score / (2 * pairs) if pairs else None
