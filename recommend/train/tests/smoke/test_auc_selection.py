from pathlib import Path

import pytest
import torch

from recommend.train.comm.model_params_validator import validate_config
from recommend.train.models.rerank.runner import NeuralRunner
from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.train_rerank import execute_train_local


@pytest.mark.parametrize("mode,expected", [("max", 5), ("min", 2)])
def test_metric_selection_without_early_stop(tmp_path, monkeypatch, mode, expected):
    config = make_experiment(tmp_path, "deepfm")
    config["max_epochs"] = 6
    config["early_stopping_patience"] = None
    config["selection"] = {"mode": mode, "task": "completion", "metric": "auc"}
    original = NeuralRunner.evaluate
    scores = iter([0.7, 0.6, None, 0.65, 0.8, 0.75, 0.8])
    states = []

    def evaluate(self, arrays):
        states.append({k: v.clone() for k, v in self.model.state_dict().items()})
        result = original(self, arrays)
        result["completion"]["metrics"]["auc"]["value"] = next(scores)
        return result

    monkeypatch.setattr(NeuralRunner, "evaluate", evaluate)
    result = execute_train_local(config)
    import json

    root = Path(result["run_dir"])
    metrics = json.loads((root / "metrics.json").read_text())
    assert len(metrics["history"]) == 6
    assert metrics["selection"]["epoch"] == expected
    saved = torch.load(root / "checkpoint.pt", weights_only=False)
    assert all(torch.equal(v, states[expected - 1][k]) for k, v in saved["model"].items())


def test_unknown_selection_metric(tmp_path):
    config = make_experiment(tmp_path, "deepfm")
    config["selection"] = {"mode": "max", "task": "completion", "metric": "missing"}
    with pytest.raises(ValueError, match="unknown task or metric"):
        validate_config(config)
