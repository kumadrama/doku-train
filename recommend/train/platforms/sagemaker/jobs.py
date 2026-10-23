from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from recommend.train.platforms.sagemaker.artifacts import verify_archive
from recommend.train.platforms.sagemaker.config import Binding, split_s3


def fetch_result(sm: Any, s3: Any, name: str, binding: Binding) -> dict[str, Any]:
    job = sm.describe_training_job(TrainingJobName=name)
    if job["TrainingJobStatus"] != "Completed":
        raise ValueError("job is not completed; uploaded artifacts alone are not success")
    uri = job["ModelArtifacts"]["S3ModelArtifacts"]
    if not uri.startswith(binding.experiment_uri + "runs/"):
        raise ValueError("unexpected artifact location")
    bucket, key = split_s3(uri)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "result.tar.gz"
        s3.download_file(bucket, key, str(archive))
        dest = Path(tmp) / "result"
        dest.mkdir()
        with tarfile.open(archive) as tar:
            total = 0
            for member in tar:
                path = PurePosixPath(member.name)
                total += member.size
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                    or total > 10 * 1024**3
                ):
                    raise ValueError("unsafe or oversized artifact archive")
                tar.extract(member, dest, filter="data")
        verify_archive(dest)
        package_bytes = (dest / "submission.json").read_bytes()
        if hashlib.sha256(package_bytes).hexdigest() != job.get("Environment", {}).get(
            "DOKU_SUBMISSION_SHA256"
        ):
            raise ValueError("artifact belongs to a different submission")
        return {
            "job": name,
            "status": "complete",
            "training_seconds": job.get("TrainingTimeInSeconds"),
            "billable_seconds": job.get("BillableTimeInSeconds"),
            "lineage": json.loads((dest / "lineage.json").read_text()),
            "validation": json.loads((dest / "metrics.json").read_text())["validation"],
            "test": (
                json.loads((dest / "test-metrics.json").read_text())
                if "test" in json.loads(package_bytes)["runtime"].get("stages", ["train", "test"])
                else None
            ),
        }


def sync_events(s3: Any, binding: Binding, run_id: str, destination: Path) -> None:
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in run_id):
        raise ValueError("invalid run ID")
    bucket, prefix = split_s3(binding.experiment_uri + f"runs/{run_id}/tensorboard/")
    target = destination / run_id
    target.mkdir(parents=True, exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            name = item["Key"][len(prefix) :]
            if "/" in name or not name.startswith("events.out.tfevents."):
                raise ValueError("unexpected TensorBoard object")
            path = target / name
            if not path.exists():
                partial = target / (name + ".partial")
                s3.download_file(bucket, item["Key"], str(partial))
                partial.replace(path)


def view_events(
    s3: Any, binding: Binding, run_ids: list[str], directory: Path, interval: int, port: int
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        sync_events(s3, binding, run_id, directory)
    args = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(directory),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--purge_orphaned_data",
        "false",
    ]
    with subprocess.Popen(args) as process:
        print(
            f"TensorBoard: http://127.0.0.1:{port} (closing this viewer does not stop training)",
            flush=True,
        )
        try:
            while process.poll() is None:
                time.sleep(interval)
                for run_id in run_ids:
                    sync_events(s3, binding, run_id, directory)
            if process.returncode:
                raise RuntimeError(f"TensorBoard exited: {process.returncode}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
