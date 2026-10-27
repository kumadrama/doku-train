from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from recommend.train.comm.training_pipeline import test_run as evaluate_run
from recommend.train.models.rerank.runner import NeuralRunner
from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.train_rerank import execute_train_local


def test_timing_preserves_model_identity_metrics_and_repeated_test(tmp_path, capsys):
    config = make_experiment(tmp_path, "deepfm", rows_per_day=8)
    config["execution"].update(reader_workers=2, s3_transfer_threads=2, prefetch_batches=2)
    off = execute_train_local(config)
    off_root = Path(off["run_dir"])
    assert not (off_root / "timing").exists()
    config["output"]["run_id"] = "timed"
    config["profiling"] = {"enabled": True, "print_logs": False}
    on = execute_train_local(config)
    on_root = Path(on["run_dir"])
    assert "DOKU_TIMING" not in capsys.readouterr().out
    for key in ["history", "selection", "validation"]:
        assert off["metrics"][key] == on["metrics"][key]
    a = torch.load(off_root / "checkpoint.pt", weights_only=False)
    b = torch.load(on_root / "checkpoint.pt", weights_only=False)
    assert a["identity"] == b["identity"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    summary = json.loads(next((on_root / "timing").glob("train-*-summary.json")).read_text())
    assert summary["status"] == "complete"
    assert "train/fit_features/shard_checksum" in summary["stages"]
    assert "train/encode_validation/width_scan/shard_checksum" in summary["stages"]
    assert "shard_checksum" not in summary["stages"]
    assert "train/encode_train/width_scan" not in summary["stages"]
    names = {k.split("/")[-1] for k in summary["stages"]}
    assert {
        "freeze_input",
        "fit_features",
        "width_scan",
        "batch_encode",
        "array_assign",
        "parquet_batch",
        "column_decode",
        "row_validate",
        "batch_transform",
        "vocab_collect",
        "vocab_finalize",
        "shard_checksum",
        "model_initialize",
        "batch_fetch",
        "forward",
        "loss",
        "backward",
        "optimizer",
        "metrics",
        "validation_epoch",
        "validation_selected",
        "resume_save",
        "checkpoint_serialize",
        "checkpoint_write",
    } <= names
    baseline = evaluate_run(off_root, NeuralRunner)
    assert evaluate_run(on_root, NeuralRunner) == baseline
    assert evaluate_run(on_root, NeuralRunner) == baseline
    assert len(list((on_root / "timing").glob("test-*-summary.json"))) == 2
    assert len(list((on_root / "timing").glob("train-*-summary.json"))) == 1
    # Explicit false remains semantically identical to omission.
    config["output"]["run_id"] = "explicit-off"
    config["profiling"] = {"enabled": False, "print_logs": False}
    again = execute_train_local(config)
    assert not (Path(again["run_dir"]) / "timing").exists()


def test_failure_keeps_timing_and_does_not_claim_success(tmp_path, monkeypatch):
    config = make_experiment(tmp_path, "deepfm", rows_per_day=8)
    config["profiling"] = {"enabled": True}

    def fail(self, arrays):
        raise RuntimeError("synthetic evaluation error")

    monkeypatch.setattr(NeuralRunner, "evaluate", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        execute_train_local(config)
    root = Path(config["output"]["root"]) / config["output"]["run_id"]
    summary = json.loads(next((root / "timing").glob("*-summary.json")).read_text())
    assert summary["status"] == "failed"
    assert "train/fit_model/validation_epoch" in summary["stages"]
    assert not (root / "status.json").exists()
    assert (root / "failure.json").exists()


def test_resume_allows_switching_profiling(tmp_path, monkeypatch):
    config = make_experiment(tmp_path, "deepfm", rows_per_day=8)
    config["max_epochs"] = 3
    original = NeuralRunner.evaluate
    calls = 0

    def interrupt(self, arrays):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(self, arrays)

    monkeypatch.setattr(NeuralRunner, "evaluate", interrupt)
    with pytest.raises(RuntimeError):
        execute_train_local(config)
    root = Path(config["output"]["root"]) / config["output"]["run_id"]
    monkeypatch.setattr(NeuralRunner, "evaluate", original)
    config["output"]["run_id"] = "resumed"
    config["profiling"] = {"enabled": True}
    resumed = execute_train_local(config, resume=root / "resume.pt")
    full_config = copy.deepcopy(config)
    full_config["output"]["run_id"] = "full"
    full = execute_train_local(full_config)
    a = torch.load(Path(resumed["run_dir"]) / "checkpoint.pt", weights_only=False)
    b = torch.load(Path(full["run_dir"]) / "checkpoint.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
