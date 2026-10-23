"""Optional pretraining orchestration; concrete runners/exporters are injected."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recommend.train.comm.model_params_validator import load_config, validate_config
from recommend.train.comm.tree_buckets import validate_buckets


def load_tree_config(config: dict[str, Any]) -> dict[str, Any] | None:
    if not config.get("tree_binning", {}).get("enabled"):
        return None
    teacher = load_config(Path(config["tree_binning"]["config_path"]))
    validate_tree_config(config, teacher)
    return teacher


def validate_tree_config(config: dict[str, Any], teacher: dict[str, Any]) -> None:
    validate_config(teacher)
    if teacher["model_type"] != "rerank_tree":
        raise ValueError("tree_binning config_path must reference a tree YAML")
    if teacher["dataset"] != config["dataset"] or teacher["split"] != config["split"]:
        raise ValueError("tree binning requires the same dataset and date splits")


def prepare_tree_buckets(
    config: dict[str, Any],
    root: Path,
    resume: Path | None,
    teacher: dict[str, Any] | None,
    train_tree: Callable[..., dict[str, Any]],
    export: Callable[[Path], dict[str, Any]],
    local_root: Path | None,
) -> dict[str, Any]:
    if resume is not None:
        state = json.loads((resume.parent / "fitted-feature-state/state.json").read_text())
        artifact: dict[str, Any] = state["tree_buckets"]
    else:
        if teacher is None:
            raise ValueError("tree training YAML was not loaded")
        validate_tree_config(config, teacher)
        teacher = copy.deepcopy(teacher)
        teacher["output"] = {"root": str(root / "tree-training"), "run_id": "teacher"}
        teacher["execution"]["cache_dir"] = str(root / "tree-cache")
        result = train_tree(teacher, local_root=local_root, frozen_input=root / "input")
        artifact = export(Path(result["run_dir"]))
    validate_buckets(artifact)
    with (root / "tree-buckets.json").open("x") as f:
        json.dump(artifact, f, sort_keys=True, allow_nan=False, indent=2)
    return artifact
