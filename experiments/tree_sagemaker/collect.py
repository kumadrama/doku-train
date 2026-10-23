"""Collect verified artifacts and tabulate validation metrics without selecting features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import boto3

from recommend.train.platforms.sagemaker.artifacts import verify_archive
from recommend.train.platforms.sagemaker.config import load_runtime, split_s3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    _, binding = load_runtime(args.runtime)
    session = boto3.Session(profile_name=args.profile, region_name=binding.region)
    if session.client("sts").get_caller_identity()["Account"] != binding.account_id:
        raise ValueError("wrong account")
    sm, s3 = session.client("sagemaker"), session.client("s3")
    rows = []
    for name, record in json.loads(args.state.read_text()).items():
        if record["status"] != "Completed":
            continue
        job = sm.describe_training_job(TrainingJobName=record["job"])
        if job["TrainingJobStatus"] != "Completed":
            raise ValueError("job no longer completed")
        uri = job["ModelArtifacts"]["S3ModelArtifacts"]
        if not uri.startswith(binding.experiment_uri + "runs/"):
            raise ValueError("unexpected artifact prefix")
        dest = args.destination / name
        if not dest.exists():
            dest.mkdir(parents=True)
            archive = dest / "model.tar.gz"
            bucket, key = split_s3(uri)
            s3.download_file(bucket, key, str(archive))
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                if sum(m.size for m in members) > 10 * 1024**3:
                    raise ValueError("oversized archive")
                for m in members:
                    p = PurePosixPath(m.name)
                    if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                        raise ValueError("unsafe archive")
                tar.extractall(dest, members=members, filter="data")
        verify_archive(dest)
        if (
            hashlib.sha256((dest / "submission.json").read_bytes()).hexdigest()
            != job["Environment"]["DOKU_SUBMISSION_SHA256"]
        ):
            raise ValueError("submission identity mismatch")
        metrics = json.loads((dest / "metrics.json").read_text())
        for task, values in metrics["validation"].items():
            for metric, value in values["metrics"].items():
                rows.append(
                    {
                        "experiment": name,
                        "task": task,
                        "metric": metric,
                        "value": value.get("value") if isinstance(value, dict) else value,
                        "details": json.dumps(value),
                        "selected_rounds": metrics["selection"]["rounds"][task],
                        "training_seconds": job.get("TrainingTimeInSeconds"),
                        "artifact_uri": uri,
                    }
                )
    if rows:
        with (args.destination / "validation.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("verified metric rows", len(rows))


if __name__ == "__main__":
    main()
