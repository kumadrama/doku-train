"""Tree-only evaluation loss; independent of the neural loss implementation."""

import numpy as np


def logloss(margin: np.ndarray, labels: np.ndarray) -> float:
    """Stable binary cross entropy from raw XGBoost margins."""
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels must be zero or one")
    raw = margin[:, 0].astype(np.float64)
    return float((np.maximum(raw, 0) - raw * labels + np.log1p(np.exp(-np.abs(raw)))).mean())
