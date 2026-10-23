from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from recommend.train.models.rerank.runner import NeuralRunner
from recommend.train.tests.tree_binning_fixture import FIELD, make_tree_experiment
from recommend.train.train_rerank import execute_test, execute_train, execute_train_local


def test_tree_then_nn_and_test_without_yaml(tmp_path):
    config = make_tree_experiment(tmp_path)
    result = execute_train(config)
    root = Path(result["run_dir"])
    artifact = json.loads((root / "tree-buckets.json").read_text())
    teacher = root / "tree-training/teacher"
    assert artifact == json.loads((teacher / "tree-buckets.json").read_text())
    assert FIELD in artifact["features"]
    assert set(artifact["source"]["tasks"]) == set(artifact["source"]["config"]["tasks"])
    assert (root / "input/manifest.json").read_bytes() == (
        teacher / "input/manifest.json"
    ).read_bytes()
    assert not (teacher / "test-metrics.json").exists()
    state = json.loads((root / "fitted-feature-state/state.json").read_text())
    assert state["tree_buckets"] == artifact
    assert state["vocabs"][FIELD + "_tree"] == list(range(len(artifact["features"][FIELD]) + 1))
    (tmp_path / "teacher.yaml").unlink()
    assert execute_test(root)["completion"]["metrics"]["auc"]["value"] is not None


def test_tree_bucket_resume_does_not_retrain(tmp_path, monkeypatch):
    config = make_tree_experiment(tmp_path)
    original = NeuralRunner.evaluate
    calls = 0

    def interrupt(self, arrays):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupt second epoch")
        return original(self, arrays)

    monkeypatch.setattr(NeuralRunner, "evaluate", interrupt)
    with pytest.raises(RuntimeError, match="interrupt second"):
        execute_train_local(config)
    first = Path(config["output"]["root"]) / config["output"]["run_id"]
    assert not (first / "status.json").exists()
    monkeypatch.setattr(NeuralRunner, "evaluate", original)
    teacher_yaml = tmp_path / "teacher.yaml"
    saved = teacher_yaml.read_text()
    teacher_yaml.unlink()
    config["output"]["run_id"] = "resumed"
    resumed = Path(execute_train_local(config, resume=first / "resume.pt")["run_dir"])
    assert not (resumed / "tree-training").exists()
    teacher_yaml.write_text(saved)
    config["output"]["run_id"] = "uninterrupted"
    uninterrupted = Path(execute_train_local(config)["run_dir"])
    a = torch.load(resumed / "checkpoint.pt", weights_only=False)
    b = torch.load(uninterrupted / "checkpoint.pt", weights_only=False)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
