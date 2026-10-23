from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import boto3

from recommend.train.comm import metric_events
from recommend.train.comm.training_pipeline import source_identity, test_run, train
from recommend.train.platforms.sagemaker.artifacts import archive_run
from recommend.train.platforms.sagemaker.config import (
    Binding,
    Runtime,
    allowed,
    validate_experiment,
)
from recommend.train.platforms.sagemaker.events import EventSink
from recommend.train.platforms.sagemaker.resources import ResourceSampler
from recommend.train.train_rerank import model_factory


def run_package(package_path: Path, root: Path, s3: Any, *, expected_sha: str) -> None:
    payload = package_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("submission checksum mismatch")
    package = json.loads(payload)
    if package["version"] != "doku-sagemaker-submission/v1":
        raise ValueError("unsupported package")
    runtime = Runtime.model_validate(package["runtime"])
    binding = Binding.model_validate(package["binding"])
    source = source_identity()
    for field, expected in [
        ("revision", runtime.source_revision),
        ("working_source_sha256", runtime.source_sha256),
        ("lock_sha256", runtime.lock_sha256),
    ]:
        if source[field] != expected:
            raise ValueError(f"image {field} differs from submitted identity")
    config = copy.deepcopy(package["config"])
    validate_experiment(config, runtime)
    allowed(
        config["dataset"]["manifest_uri"],
        [*binding.input_prefixes, binding.experiment_uri + "submissions/"],
    )
    frozen = root / "work" / "frozen-input"
    frozen.mkdir(parents=True, exist_ok=False)
    expected_names = {
        "manifest.json",
        "record.json",
        "contracts/features.json",
        "contracts/labels.json",
    }
    if set(package["input"]) != expected_names:
        raise ValueError("invalid frozen input file set")
    for name, content in package["input"].items():
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for partition in json.loads((frozen / "manifest.json").read_text())["partitions"]:
        allowed(partition["uri"], binding.input_prefixes)
    run_id = package["run_id"]
    config["output"] = {"root": str(root / "work" / "runs"), "run_id": run_id}
    config["execution"]["cache_dir"] = str(root / "work" / "cache")
    diagnostic = root / "output" / "data"
    diagnostic.mkdir(parents=True, exist_ok=True)
    sink = EventSink(
        root / "work" / "events",
        s3,
        binding.experiment_uri + f"runs/{run_id}/tensorboard/",
        binding.kms_key_arn,
        runtime.sync_interval_seconds,
    )
    stage_times = {}
    sampler = ResourceSampler(root)
    try:
        with metric_events.observe(sink, runtime.loss_interval_steps):
            print("DOKU_STAGE train starting", flush=True)
            start = time.monotonic()
            result = train(config, model_factory(config["model_type"]), frozen_input=frozen)
            stage_times["train"] = time.monotonic() - start
            run_dir = Path(result["run_dir"])
            if "test" in runtime.stages:
                print("DOKU_STAGE test starting", flush=True)
                start = time.monotonic()
                test = test_run(run_dir, model_factory(config["model_type"]))
                selection = result["metrics"]["selection"]
                metric_events.evaluation(
                    "test", selection.get("global_step", 0), selection.get("epoch") or 0, test
                )
                stage_times["test"] = time.monotonic() - start
        sink.close()
        print("DOKU_STAGE archive starting", flush=True)
        archive_run(run_dir, root / "model", package_path)
    finally:
        report = {
            **sampler.close(),
            "stage_seconds": stage_times,
            "disk_free_bytes": shutil.disk_usage(root).free,
            "working_files_bytes": sum(
                p.stat().st_size for p in (root / "work").rglob("*") if p.is_file()
            ),
        }
        (diagnostic / "resources.json").write_text(json.dumps(report))
        if sink.thread.is_alive():
            sink.close()


def main() -> None:
    root = Path("/opt/ml")
    try:
        if sys.argv[1:] != ["train"]:
            raise ValueError("SageMaker train argument required")
        package = root / "input/data/submission/package.json"
        binding = Binding.model_validate(json.loads(package.read_text())["binding"])
        s3 = boto3.client("s3", region_name=binding.region)
        run_package(package, root, s3, expected_sha=os.environ["DOKU_SUBMISSION_SHA256"])
    except Exception as exc:
        failure = root / "output/failure"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(f"{type(exc).__name__}: {exc}"[:1024])
        raise


if __name__ == "__main__":
    main()
