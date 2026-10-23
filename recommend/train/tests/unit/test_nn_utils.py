"""Shared loss numerical, gradient and evaluation contracts."""

import numpy as np
import pytest
import torch

from recommend.train.comm.nn_utils import get_loss
from recommend.train.comm.task_metrics import metric_values


def test_extreme_logits_and_weighted_gradients():
    logits = torch.tensor([[-1000.0], [1000.0]], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    weights = torch.tensor([0.0, 2.0])
    loss = get_loss({"loss_type": "logloss"}, logits, labels, loss_weight_data=weights)
    assert loss.item() == 1000.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.0], [1.0]]))


def test_metric_uses_own_labels_and_mask_with_raw_margin():
    arrays = {
        "metric:t": np.zeros((3, 1)),
        "value:t:logloss": np.array([[1.0], [0.0], [1.0]]),
        "mask:t": np.ones((3, 1), dtype=bool),
        "valid:t:logloss": np.array([[True], [False], [True]]),
    }
    task = {"metrics": {"logloss": {"func": "logloss"}}}
    margin = np.array([-1000.0, 20.0, 0.0])
    result = metric_values(task, "t", arrays, np.array([0.0, 1.0, 0.5]), logits=margin)
    expected = get_loss(
        {"loss_type": "logloss"},
        torch.tensor([[-1000.0], [0.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    assert result["logloss"]["value"] == expected.item()
    assert result["logloss"]["valid_rows"] == 2
    with pytest.raises(ValueError, match="raw logits"):
        metric_values(task, "t", arrays, np.array([0.0, 1.0, 0.5]))
    arrays["valid:t:logloss"][:] = False
    empty = metric_values(task, "t", arrays, np.array([0.0, 1.0, 0.5]), logits=margin)
    assert empty["logloss"]["value"] is None
    assert empty["logloss"]["reason"] == "no_valid_samples"


def test_tree_loss_is_independent_and_matches_binary_objective():
    from recommend.train.models.rerank_tree.loss_utils import logloss

    raw = np.array([[-1000.0], [1000.0], [0.0]])
    labels = np.array([1.0, 0.0, 1.0])
    assert logloss(raw, labels) == pytest.approx((2000 + np.log(2)) / 3)
    arrays = {
        "metric:t": labels[:, None],
        "mask:t": np.ones((3, 1), dtype=bool),
        "valid:t:logloss": np.ones((3, 1), dtype=bool),
    }
    task = {"metrics": {"logloss": {"func": "logloss"}}}
    from unittest.mock import patch

    with patch("recommend.train.comm.nn_utils.get_loss", side_effect=AssertionError):
        result = metric_values(
            task, "t", arrays, np.array([0.0, 1.0, 0.5]), logits=raw, logloss_func=logloss
        )
    assert result["logloss"]["value"] == logloss(raw, labels)


def test_tree_import_does_not_load_torch():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import recommend.train.models.rerank_tree.runner; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )
