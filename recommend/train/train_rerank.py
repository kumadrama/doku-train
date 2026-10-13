"""Composition root with process isolation for independent native CPU runtimes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from recommend.train.comm.training_pipeline import train
from recommend.train.eval_rerank import execute_test as execute_test
from recommend.train.eval_rerank import execute_test_local as execute_test_local


def model_factory(name: str) -> Any:
    if name == "rerank":
        from recommend.train.models.rerank.runner import NeuralRunner

        return NeuralRunner
    if name == "rerank_tree":
        from recommend.train.models.rerank_tree.runner import TreeRunner

        return TreeRunner
    raise ValueError("unknown model plugin")


def execute_train_local(
    config: dict[str, Any], *, local_root: Path | None = None, resume: Path | None = None
) -> dict[str, Any]:
    if resume is not None and config["model_type"] == "rerank_tree":
        raise ValueError("XGBoost partial training resume is unsupported")
    return train(config, model_factory(config["model_type"]), local_root=local_root, resume=resume)


def execute_train(
    config: dict[str, Any], *, local_root: Path | None = None, resume: Path | None = None
) -> dict[str, Any]:
    root = Path(config["output"]["root"]) / config["output"]["run_id"]
    if root.exists():
        raise FileExistsError(str(root))
    with tempfile.TemporaryDirectory(prefix="doku-trial-") as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        args = [sys.executable, "-m", "recommend.train.run", "train", "--config", str(path)]
        if local_root:
            args += ["--local-root", str(local_root)]
        if resume:
            args += ["--resume", str(resume)]
        completed = subprocess.run(args, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"training process failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        return {
            "status": "complete",
            "run_dir": str(root.resolve()),
            "metrics": json.loads((root / "metrics.json").read_text()),
        }
