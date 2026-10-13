from __future__ import annotations

import json

import pytest

from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.train_rerank import execute_test, execute_train


@pytest.mark.parametrize("model", ["dnn", "deepfm", "xgboost"])
def test_new_cpu_training(tmp_path, model):
    config = make_experiment(tmp_path, model)
    result = execute_train(config)
    assert result["status"] == "complete"
    from pathlib import Path

    root = Path(result["run_dir"])
    assert not (root / "test-metrics.json").exists()
    metrics = execute_test(root)
    assert metrics["watch_seconds"]["metrics"]["mse"]["value"] >= 0
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        execute_train(config)


@pytest.mark.parametrize("keep_probability", [1.0, 0.9])
def test_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, keep_probability):
    from pathlib import Path

    import torch

    from recommend.train.models.rerank.runner import NeuralRunner
    from recommend.train.train_rerank import execute_train_local

    config = make_experiment(tmp_path, "dnn")
    config["output_layer"]["layer_dropout"] = keep_probability
    config["tasks"]["completion"]["layer_dropout"] = keep_probability
    original = NeuralRunner.evaluate
    calls = 0

    def interrupted(self, arrays):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(self, arrays)

    monkeypatch.setattr(NeuralRunner, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        execute_train_local(config)
    first = Path(config["output"]["root"]) / config["output"]["run_id"]
    assert not (first / "status.json").exists()
    monkeypatch.setattr(NeuralRunner, "evaluate", original)
    config["output"]["run_id"] = "resumed"
    resumed = execute_train(config, resume=first / "resume.pt")
    config["output"]["run_id"] = "uninterrupted"
    uninterrupted = execute_train(config)
    a = torch.load(Path(resumed["run_dir"]) / "checkpoint.pt", weights_only=False)
    b = torch.load(Path(uninterrupted["run_dir"]) / "checkpoint.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key


def test_full_nine_trial_search(tmp_path):
    import yaml

    from recommend.train.comm.search import execute_search
    from recommend.train.tests.synthetic_config import synthetic_search

    root = tmp_path / "fixture"
    for model in ["dnn", "deepfm", "xgboost"]:
        make_experiment(root, model, rows_per_day=8)
    config = synthetic_search()
    # Tiny fixture budget, same 4+4+1 grid cardinality and identities.
    config["search"]["fixed"]["batch_size"] = 32
    (root / "search.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = execute_search(root / "search.yaml", tmp_path / "search", execute_train)
    assert len(report["results"]) == 9
    assert all(r["status"] == "complete" for r in report["results"].values()), report
    assert not report["test_evaluated"]
    assert len(report["comparison"]) == 9


def test_published_identity_is_not_a_fixture_only_gate(tmp_path):
    from pathlib import Path

    from recommend.train.comm.datasvr.published_dataset import digest

    config = make_experiment(tmp_path, "xgboost", rows_per_day=8)
    path = Path(config["dataset"]["manifest_uri"].removeprefix("file://"))
    manifest = json.loads(path.read_text())
    manifest["dataset_id"] = "fyp-rank-train/contract-test"
    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    path.write_text(json.dumps(manifest))
    result = execute_train(config)
    root = Path(result["run_dir"])
    assert execute_test(root)["watch_seconds"]["metrics"]["mse"]["value"] >= 0
