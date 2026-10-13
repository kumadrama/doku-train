"""Export successful tree training outputs without importing training model internals."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.comm.publication import encode
from recommend.train.comm.runtime_budget import MemoryBudget, check_memory


def gate(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    rules = config.get("publication")
    if not rules or set(rules) != set(config["tasks"]):
        raise ValueError("publication rules required for all tasks")
    decisions = {}
    for task, rule in rules.items():
        record = metrics.get(task, {}).get("metrics", {}).get(rule["metric"], {})
        value = record.get("value")
        passed = (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and np.isfinite(value)
            and rule["minimum"] <= value <= 1
        )
        decisions[task] = {
            "passed": bool(passed),
            "metric": rule["metric"],
            "value": value,
            "minimum": rule["minimum"],
            "reason": record.get("reason"),
        }
    return {"passed": all(d["passed"] for d in decisions.values()), "tasks": decisions}


def _export_model(run_dir: Path, output: Path, *, version: str | None = None) -> Path:
    config = validate_config(json.loads((run_dir / "resolved-config.json").read_text()))
    if config["model_type"] != "rerank_tree":
        raise ValueError("only CPU tree export is implemented")
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("export requires successful training")
    lineage = json.loads((run_dir / "lineage.json").read_text())
    check_memory((run_dir / "checkpoint.xgb.zip").stat().st_size * 4)
    payload = (run_dir / "checkpoint.xgb.zip").read_bytes()
    if hashlib.sha256(payload).hexdigest() != lineage["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    state = json.loads((run_dir / "fitted-feature-state/state.json").read_text())
    if digest(state) != lineage["feature_state_sha256"]:
        raise ValueError("feature state checksum mismatch")
    metrics = json.loads((run_dir / "test-metrics.json").read_text())
    evidence = json.loads((run_dir / "test-evidence.json").read_text())
    if evidence != {
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "config_sha256": digest(config),
        "metrics_sha256": digest(metrics),
    }:
        raise ValueError("test evidence does not match training output")
    decision = gate(config, metrics)
    if not decision["passed"]:
        raise ValueError("publication gate rejected")
    import xgboost as xgb

    version = version or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", version):
        raise ValueError("invalid version")
    target: Path = output / str(config["model_name"]) / str(config["split"]["test_to"]) / version
    target.mkdir(parents=True, exist_ok=False)
    files: dict[str, bytes] = {}
    numeric = config["tree_inputs"]["numeric"]
    categorical = config["tree_inputs"]["categorical"]
    columns = numeric + categorical
    types = ["q"] * len(numeric) + ["c"] * len(categorical)
    # Public encoded-input probes; never ship real user samples.
    probe = np.zeros((8, len(columns)), dtype=np.float32)
    for i in range(len(numeric)):
        probe[:, i] = np.linspace(-1, 2, 8)
    matrix = xgb.DMatrix(probe, feature_types=types, enable_categorical=True, nthread=1)
    tasks: dict[str, Any] = {}
    with zipfile.ZipFile(run_dir / "checkpoint.xgb.zip") as archive:
        check_memory(sum(i.file_size for i in archive.infolist()) * 4)
        meta = json.loads(archive.read("metadata.json"))
        expected = {"metadata.json"} | {f"{i}.ubj" for i in range(len(config["tasks"]))}
        if (
            meta["format"] != "rerank-tree/v2"
            or set(meta["tasks"]) != set(config["tasks"])
            or len(meta["tasks"]) != len(config["tasks"])
            or set(archive.namelist()) != expected
            or len(archive.namelist()) != len(expected)
        ):
            raise ValueError("invalid tree checkpoint contract")
        if meta["xgboost_version"] != xgb.__version__:
            raise ValueError("export runtime must match training XGBoost version")
        for i, name in enumerate(meta["tasks"]):
            task = config["tasks"][name]
            original = xgb.Booster()
            original.load_model(bytearray(archive.read(f"{i}.ubj")))
            original.set_param({"device": "cpu", "nthread": 1})
            if original.num_features() != len(columns):
                raise ValueError("input contract width mismatch")
            model = bytes(original.save_raw(raw_format="ubj"))
            restored = xgb.Booster()
            restored.load_model(bytearray(model))
            restored.set_param({"device": "cpu", "nthread": 1})

            def predict(booster: xgb.Booster, fn: str = task["predict_fn"]) -> np.ndarray:
                margin = booster.predict(matrix, output_margin=True)
                return np.exp(-np.logaddexp(0, -margin)) if fn == "sigmoid" else margin

            np.testing.assert_allclose(predict(original), predict(restored), rtol=1e-6, atol=1e-7)
            filename = f"models/{i}.ubj"
            files[filename] = model
            tasks[name] = {
                "file": filename,
                "objective": task["objective"],
                "prediction_input": "raw_margin",
                "unit": task["output_unit"],
                "predict_fn": task["predict_fn"],
                "label_expression": task["label"],
                "metric_label": task.get("metric_label", task["label"]),
            }
    contract = {
        "format": "doku-tree-features/v1",
        "columns": columns,
        "types": types,
        "encoding": {
            key: config[key]
            for key in [
                "features",
                "feature_default_values",
                "feature_transformations",
                "hist_options",
                "input_features",
                "tree_inputs",
            ]
        },
    }
    files.update(
        {
            "feature-contract.json": encode(contract),
            "fitted-feature-state/state.json": encode(state),
            "metrics.json": encode({"test": metrics, "gate": decision}),
            "lineage.json": encode(lineage),
        }
    )
    manifest = {
        "format": "doku-tree-artifact/v1",
        "model_type": config["model_type"],
        "model_name": config["model_name"],
        "model_version": version,
        "date": config["split"]["test_to"],
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": config["output"]["run_id"],
        "config_sha256": digest(config),
        "xgboost_version": xgb.__version__,
        "tasks": tasks,
        "parity": {
            "status": "passed",
            "kind": "synthetic_encoded_cpu",
            "rows": 8,
            "rtol": 1e-6,
            "atol": 1e-7,
        },
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }
    for name, body in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    check_memory()
    (target / "manifest.json").write_bytes(encode(manifest))
    return target


def export_model(run_dir: Path, output: Path, *, version: str | None = None) -> Path:
    config = json.loads((run_dir / "resolved-config.json").read_text())
    budget = MemoryBudget(config["execution"]["memory_limit_bytes"])
    try:
        budget.start()
        from recommend.train.comm.run_lock import run_lock

        with run_lock(run_dir):
            return _export_model(run_dir, output, version=version)
    finally:
        budget.close()


def execute_export(run_dir: Path, output: Path, *, version: str) -> Path:
    """Isolate native XGBoost from a caller that may already have loaded PyTorch."""
    import subprocess
    import sys

    args = [
        sys.executable,
        "-m",
        "recommend.train.run",
        "export",
        "--run-dir",
        str(run_dir),
        "--output",
        str(output),
        "--version",
        version,
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"export process failed: {result.stderr[-2000:]}")
    return Path(result.stdout.strip())
