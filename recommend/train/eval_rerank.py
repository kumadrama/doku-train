"""Independent evaluation use case; never trains or modifies a checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from recommend.train.comm.process_logs import run_captured
from recommend.train.comm.training_pipeline import test_run


def execute_test_local(run_dir: Path, *, local_root: Path | None = None) -> dict[str, Any]:
    from recommend.train.train_rerank import feature_plan_factory, model_factory

    config = json.loads((run_dir / "resolved-config.json").read_text())
    return test_run(
        run_dir,
        model_factory(config["model_type"]),
        local_root=local_root,
        plan_factory=feature_plan_factory(config["model_type"]),
    )


def execute_test(run_dir: Path, *, local_root: Path | None = None) -> dict[str, Any]:
    args = [sys.executable, "-m", "recommend.train.run", "test", "--run-dir", str(run_dir)]
    if local_root:
        args += ["--local-root", str(local_root)]
    config = json.loads((run_dir / "resolved-config.json").read_text())
    completed = run_captured(args, config)
    if completed.returncode:
        raise RuntimeError(f"evaluation process failed: {completed.stderr[-2000:]}")
    result: dict[str, Any] = json.loads((run_dir / "test-metrics.json").read_text())
    return result
