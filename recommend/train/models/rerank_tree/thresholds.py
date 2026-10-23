"""Export all selected boosters' numeric split thresholds by declared feature name."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Any

import numpy as np
import xgboost as xgb

from recommend.train.comm.tree_buckets import VERSION, validate_buckets


def export_thresholds(
    payload: bytes, config: dict[str, Any], lineage: dict[str, Any]
) -> dict[str, Any]:
    if config["model_type"] != "rerank_tree":
        raise ValueError("tree bucket export requires a tree model")
    if hashlib.sha256(payload).hexdigest() != lineage["checkpoint_sha256"]:
        raise ValueError("tree checkpoint digest mismatch")
    numeric = config["tree_inputs"]["numeric"]
    cuts: dict[str, set[float]] = {name: set() for name in numeric}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if set(metadata["tasks"]) != set(config["tasks"]):
            raise ValueError("tree checkpoint task mismatch")
        for i, _ in enumerate(metadata["tasks"]):
            booster = xgb.Booster()
            booster.load_model(bytearray(archive.read(f"{i}.ubj")))
            model = json.loads(booster.save_raw(raw_format="json"))
            for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
                for node, left in enumerate(tree["left_children"]):
                    index = tree["split_indices"][node]
                    if left == -1 or tree["split_type"][node] != 0 or index >= len(numeric):
                        continue
                    cuts[numeric[index]].add(float(np.float32(tree["split_conditions"][node])))
    source_config = {key: value for key, value in config.items() if key != "output"}
    source_config["execution"] = {
        key: value for key, value in config["execution"].items() if key != "cache_dir"
    }
    artifact = {
        "version": VERSION,
        "comparison": "float32_right",
        "features": {name: sorted(values) for name, values in cuts.items()},
        "source": {
            "config": source_config,
            "lineage": lineage,
            "tasks": metadata["tasks"],
            "selected_rounds": metadata["selected_rounds"],
        },
    }
    validate_buckets(artifact)
    return artifact
