"""CPU execution and artifact lifecycle, independent of model implementations."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil

from recommend.train.comm.checkpoint_agent import CheckpointAgent
from recommend.train.comm.datasvr.published_dataset import PublishedDataset, digest
from recommend.train.comm.feature_plan import FeaturePlan, prepare_arrays
from recommend.train.comm.model_params_validator import ROOT, validate_config
from recommend.train.comm.runtime_budget import MemoryBudget


class Runner(Protocol):
    checkpoint_name: str

    def fit(
        self,
        train: dict[str, np.ndarray],
        validation: dict[str, np.ndarray],
        checkpoint: CheckpointAgent,
        identity: str,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def evaluate(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]: ...
    def load(self, payload: bytes) -> None: ...


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def source_identity() -> dict[str, str]:
    root = ROOT.parents[1]
    if (root / "build-revision.txt").is_file():
        revision = (root / "build-revision.txt").read_text().strip()
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("invalid image source revision")
    else:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
    h = hashlib.sha256()
    for p in sorted((root / "recommend").rglob("*")):
        if p.is_file() and p.suffix in {".py", ".yml", ".yaml"}:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    lock = root / "uv.lock"
    return {
        "revision": revision,
        "working_source_sha256": h.hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def setup(config: dict[str, Any]) -> None:
    random.seed(config["seed"])
    np.random.seed(config["seed"])


def arrays_for(
    dataset: PublishedDataset, plan: FeaturePlan, split: str, root: Path
) -> dict[str, np.ndarray]:
    count = sum(p["rows"] for p in dataset.partitions(split))
    # Sequence widths are allocation metadata, not fitted vocabulary. Inspect only this split.
    for source in dataset.rows(split, plan.columns):
        row = plan.row(source)
        if row:
            for field in plan.tables:
                if isinstance(row[field], list):
                    plan.lengths[field] = max(plan.lengths.get(field, 1), len(row[field]))
    return prepare_arrays(
        plan,
        dataset.rows(split, plan.columns),
        root,
        count,
        max(
            0,
            dataset.config["execution"]["cache_limit_bytes"]
            - sum(p.stat().st_size for p in root.parent.rglob("*.npy")),
        ),
    )


@contextmanager
def stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    started = time.monotonic()
    print(f"DOKU_STAGE {name} starting", flush=True)
    try:
        yield
    finally:
        timings[name] = time.monotonic() - started
        print(f"DOKU_STAGE {name} seconds={timings[name]:.3f}", flush=True)


def train(
    config: dict[str, Any],
    factory: Any,
    *,
    local_root: Path | None = None,
    resume: Path | None = None,
    frozen_input: Path | None = None,
) -> dict[str, Any]:
    if resume is not None and frozen_input is not None:
        raise ValueError("resume and frozen_input are mutually exclusive")
    config = validate_config(config)
    setup(config)
    started = time.monotonic()
    root = Path(config["output"]["root"]) / config["output"]["run_id"]
    root.mkdir(parents=True, exist_ok=False)
    budget = MemoryBudget(config["execution"]["memory_limit_bytes"])
    timings: dict[str, float] = {}
    try:
        budget.start()
        with stage("freeze_input", timings):
            dataset = PublishedDataset(
                config,
                local_root=local_root,
                frozen_input=resume.parent / "input" if resume else frozen_input,
            )
            dataset.freeze(root / "input")
        config = dataset.config
        write_json(root / "resolved-config.json", config)
        plan = FeaturePlan(config, dataset.dtypes)
        with stage("fit_features", timings):
            plan.fit(dataset.rows("train", plan.columns))
        state = plan.state()
        (root / "fitted-feature-state").mkdir()
        write_json(root / "fitted-feature-state/state.json", state)
        with stage("encode_train", timings):
            train_data = arrays_for(dataset, plan, "train", root / "encoded-train")
        with stage("encode_validation", timings):
            valid_data = arrays_for(dataset, plan, "validation", root / "encoded-validation")
        source = source_identity()
        logical = {k: v for k, v in config.items() if k != "output"}
        identity = digest(
            {
                "config": logical,
                "dataset": dataset.manifest["content_sha256"],
                "features": state,
                "source": source,
            }
        )
        runner = factory(config, state)
        with stage("fit_model", timings):
            metrics = runner.fit(
                train_data,
                valid_data,
                CheckpointAgent(root),
                identity,
                CheckpointAgent.read_resume(resume) if resume else None,
            )
        metrics["resources"] = {
            "stage_seconds": timings,
            "wall_seconds": time.monotonic() - started,
            "rss_at_completion_bytes": psutil.Process().memory_info().rss,
            "sampled_peak_rss_bytes": budget.peak,
            "cpu_threads": config["execution"]["intra_op_threads"],
            "train_rows": len(next(iter(train_data.values()))),
            "validation_rows": len(next(iter(valid_data.values()))),
        }
        budget.check()
        write_json(root / "metrics.json", metrics)
        with (root / runner.checkpoint_name).open("rb") as checkpoint:
            checkpoint_digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        write_json(
            root / "lineage.json",
            {
                "identity": identity,
                **source,
                "manifest_sha256": dataset.manifest["content_sha256"],
                "contract_digests": dataset.contract_digests,
                "adapter": "doku_offline_v1",
                "manifest_uri": dataset.manifest_uri,
                "dataset_version": dataset.manifest["dataset_version"],
                "feature_schema_version": dataset.manifest["feature_schema_version"],
                "label_definition_version": dataset.manifest["label_definition_version"],
                "daily_sources": dataset.source_records,
                "maturity_status": dataset.maturity_status,
                "split": config["split"],
                "feature_state_sha256": digest(state),
                "checkpoint_sha256": checkpoint_digest,
            },
        )
        write_json(root / "status.json", {"status": "complete", "selection": metrics["selection"]})
        return {"run_dir": str(root.resolve()), "metrics": metrics, "status": "complete"}
    except Exception as e:
        write_json(
            root / "failure.json",
            {"status": "failed", "error_type": type(e).__name__, "message": str(e)},
        )
        raise
    finally:
        budget.close()


def _test_run(
    run_dir: Path, factory: Any, *, local_root: Path | None = None, encoding_dir: Path
) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("evaluation requires successful training")
    config = validate_config(json.loads((run_dir / "resolved-config.json").read_text()))
    setup(config)
    data = PublishedDataset(config, local_root=local_root, frozen_input=run_dir / "input")
    state = json.loads((run_dir / "fitted-feature-state/state.json").read_text())
    plan = FeaturePlan(config, data.dtypes)
    plan.restore(state)
    arrays = arrays_for(data, plan, "test", encoding_dir)
    runner = factory(config, state)
    lineage = json.loads((run_dir / "lineage.json").read_text())
    if digest(state) != lineage["feature_state_sha256"]:
        raise ValueError("feature state digest mismatch")
    with (run_dir / runner.checkpoint_name).open("rb") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != lineage["checkpoint_sha256"]:
            raise ValueError("checkpoint digest mismatch")
    runner.load((run_dir / runner.checkpoint_name).read_bytes())
    metrics: dict[str, Any] = runner.evaluate(arrays)
    report_tmp = run_dir / "test-metrics.partial"
    report_tmp.write_text(json.dumps(metrics, sort_keys=True, allow_nan=False))
    report_tmp.replace(run_dir / "test-metrics.json")
    (run_dir / "test-evidence.partial").unlink(missing_ok=True)
    write_json(
        run_dir / "test-evidence.partial",
        {
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "config_sha256": digest(config),
            "metrics_sha256": digest(metrics),
        },
    )
    (run_dir / "test-evidence.partial").replace(run_dir / "test-evidence.json")
    return metrics


def test_run(run_dir: Path, factory: Any, *, local_root: Path | None = None) -> dict[str, Any]:
    config = json.loads((run_dir / "resolved-config.json").read_text())
    budget = MemoryBudget(config["execution"]["memory_limit_bytes"])
    try:
        budget.start()
        from recommend.train.comm.run_lock import run_lock

        with (
            run_lock(run_dir),
            tempfile.TemporaryDirectory(prefix="evaluation-", dir=run_dir) as tmp,
        ):
            return _test_run(
                run_dir, factory, local_root=local_root, encoding_dir=Path(tmp) / "arrays"
            )
    finally:
        budget.close()
