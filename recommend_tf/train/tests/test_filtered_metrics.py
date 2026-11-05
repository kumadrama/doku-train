import numpy as np
import pytest

from recommend_tf.train.comm.eval.metrics import Metrics


def test_filters_have_independent_counts_and_auc_labels(tmp_path):
    tasks = {
        "idea": {
            "metrics": {
                "selected": {"func": "auc"},
                "reversed": {"func": "auc"},
                "error": {"func": "mae"},
                "empty": {"func": "mae"},
            }
        }
    }
    metrics = Metrics(tmp_path / "m.sqlite", tasks)
    try:
        metrics.add(
            "idea",
            [0.1, 0.9, 0.8],
            {
                "selected": {"label": [0, 1, 0], "filter": [True, True, False]},
                "reversed": {"label": [1, 0, 1], "filter": [True, True, False]},
                "error": {"label": [0, 1, 0], "filter": [False, False, True]},
                "empty": {"label": [np.nan] * 3, "filter": False},
            },
        )
        result = metrics.finish()["idea"]
        assert result["selected"] == {"value": 1.0, "reason": None, "rows": 2}
        assert result["reversed"]["value"] == 0.0
        assert result["error"] == {"value": 0.8, "reason": None, "rows": 1}
        assert result["empty"] == {"value": None, "reason": "empty", "rows": 0}
    finally:
        metrics.close()


def test_filter_rejects_numeric_mask(tmp_path):
    metrics = Metrics(tmp_path / "m.sqlite", {"x": {"metrics": {"m": {"func": "mae"}}}})
    try:
        with pytest.raises(ValueError, match="boolean"):
            metrics.add("x", [0.1], {"m": {"label": [0], "filter": [1]}})
    finally:
        metrics.close()
