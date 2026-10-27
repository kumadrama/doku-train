from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from recommend.train.comm.batch_feature_plan import BatchFeaturePlan
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.comm.training_pipeline import test_run as evaluate_run
from recommend.train.comm.training_pipeline import train
from recommend.train.models.rerank.runner import NeuralRunner
from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.train_rerank import feature_plan_factory


def test_old_and_batch_training_and_test_identical(tmp_path):
    config = make_experiment(tmp_path / "input", "deepfm", rows_per_day=13)
    config["batch_size"] = 7
    config["output"]["run_id"] = "old"
    old = train(copy.deepcopy(config), NeuralRunner, plan_factory=FeaturePlan)
    config["output"]["run_id"] = "batch"
    new = train(copy.deepcopy(config), NeuralRunner, plan_factory=BatchFeaturePlan)
    for key in ["history", "selection", "validation"]:
        assert old["metrics"][key] == new["metrics"][key]
    a, b = Path(old["run_dir"]), Path(new["run_dir"])
    for name in ["fitted-feature-state/state.json", "input/record.json"]:
        assert json.loads((a / name).read_text()) == json.loads((b / name).read_text())
    ca = torch.load(a / "checkpoint.pt", weights_only=False)
    cb = torch.load(b / "checkpoint.pt", weights_only=False)
    assert ca["identity"] == cb["identity"]
    for key in ca["model"]:
        assert torch.equal(ca["model"][key], cb["model"][key])
    assert evaluate_run(a, NeuralRunner, plan_factory=FeaturePlan) == evaluate_run(
        b, NeuralRunner, plan_factory=BatchFeaturePlan
    )
    assert feature_plan_factory("rerank_tree") is FeaturePlan
    assert feature_plan_factory("rerank") is BatchFeaturePlan
