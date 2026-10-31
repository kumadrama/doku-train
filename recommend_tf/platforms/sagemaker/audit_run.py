"""Read-only historical run audit bound to SageMaker job and original config."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from recommend_tf.train.comm.artifact_integrity import files

from .publish_model import location

ROOT_FILES = {"experiment-status.json", "experiment-config.json", "resolved-config.json"}


def audit(s3, sm, *, job_name, run_uri, source_config, outputs, output, limit, exporter=None):
    """Emit usable evidence only after provenance checks and full export parity succeed."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("positive download budget required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _audit(
            s3, sm, job_name, run_uri, Path(source_config), outputs, output, limit, exporter
        )
    except Exception as error:
        (output / "audit-report.json").write_text(
            json.dumps(
                {"status": "failed", "job_name": job_name, "error_type": type(error).__name__}
            )
        )
        raise


def _audit(s3, sm, job_name, run_uri, source_config, outputs, output, limit, exporter):
    job = sm.describe_training_job(TrainingJobName=job_name)
    env = job.get("Environment", {})
    raw_config = source_config.read_bytes()
    if (
        job.get("TrainingJobName") != job_name
        or job.get("TrainingJobStatus") != "Completed"
        or env.get("DOKU_ARTIFACT_URI", "").rstrip("/") != run_uri.rstrip("/")
        or env.get("DOKU_CONFIG_SHA256") != hashlib.sha256(raw_config).hexdigest()
        or not env.get("DOKU_SOURCE_SHA256")
        or not env.get("DOKU_LOCK_SHA256")
    ):
        raise ValueError("job provenance or original config mismatch")
    config = json.loads(raw_config)
    bucket, prefix = location(run_uri)
    prefix = prefix.rstrip("/") + "/"
    run = output / "run"
    run.mkdir()
    objects, used = {}, 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.startswith(prefix):
                raise ValueError("listed object outside run")
            name = key[len(prefix) :]
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
                raise ValueError("invalid run artifact path")
            if name not in ROOT_FILES and path.parts[0] not in {"input", "training"}:
                continue
            if name in objects:
                raise ValueError("duplicate artifact")
            response = s3.get_object(Bucket=bucket, Key=key, IfMatch=item["ETag"])
            local = run / name
            local.parent.mkdir(parents=True, exist_ok=True)
            with response["Body"] as stream, local.open("xb") as target:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    used += len(chunk)
                    if used > limit:
                        raise ValueError("audit download exceeds explicit disk budget")
                    target.write(chunk)
            objects[name] = {"etag": response["ETag"], "version_id": response.get("VersionId")}
    if not ROOT_FILES <= objects.keys():
        raise ValueError("historical run root evidence missing")
    if json.loads((run / "experiment-config.json").read_text()) != config:
        raise ValueError("run config differs from job-bound original config")
    lineage = json.loads((run / "training/lineage.json").read_text())
    if lineage["lock_digest"] != env["DOKU_LOCK_SHA256"]:
        raise ValueError("run dependency lock differs from job")
    status = json.loads((run / "experiment-status.json").read_text())
    selection = json.loads((run / "training/selection.json").read_text())
    if status.get("status") != "completed" or status.get("selected_epoch") != selection["epoch"]:
        raise ValueError("successful status and selected epoch differ")
    evidence = {"version": "doku-tf-run/v1", "files": files(run)}
    # Export owns the detailed config/state/checkpoint checks and CPU reload parity.
    if exporter is None:
        from recommend_tf.train.export_model import execute

        exporter = execute
    package = exporter(run, output / "package", outputs=outputs, evidence=evidence)
    # No source object may change during the audit; versions and ETags are retained.
    for name, observed in objects.items():
        head = s3.head_object(Bucket=bucket, Key=prefix + name)
        if head["ETag"] != observed["etag"] or head.get("VersionId") != observed["version_id"]:
            raise ValueError("historical source changed during audit")
    report = {
        "status": "verified",
        "job_name": job_name,
        "job_arn": job["TrainingJobArn"],
        "run_uri": run_uri,
        "selected_epoch": selection["epoch"],
        "validation_cutoff": config["dataset_config"]["split"]["validation_to"],
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "job_source_sha256": env["DOKU_SOURCE_SHA256"],
        "lineage_source_digest": lineage["source_digest"],
        "model_version": package["model_version"],
        "objects": objects,
    }
    (output / "audit-report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    # This completion artifact is intentionally written last; failed audits cannot bootstrap.
    (output / "evidence.json").write_text(json.dumps(evidence, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("region", "job-name", "run-uri", "source-config", "outputs", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--download-limit-bytes", type=int, required=True)
    args = parser.parse_args()
    import boto3

    session = boto3.Session(region_name=args.region)
    audit(
        session.client("s3"),
        session.client("sagemaker"),
        job_name=args.job_name,
        run_uri=args.run_uri,
        source_config=args.source_config,
        outputs=json.loads(Path(args.outputs).read_text()),
        output=args.output,
        limit=args.download_limit_bytes,
    )


if __name__ == "__main__":
    main()
