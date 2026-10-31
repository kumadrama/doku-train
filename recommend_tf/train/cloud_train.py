"""Explicit SageMaker CPU experiment entry; no production defaults."""

import hashlib
import json
import os
from pathlib import Path


def main():
    if os.environ.get("DOKU_MODE") == "verify":
        from .cloud_verify import main as verify

        return verify()
    import boto3

    from .comm.artifact_upload import ArtifactUpload
    from .experiment import execute

    path = Path(os.environ["DOKU_CONFIG"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != os.environ["DOKU_CONFIG_SHA256"]:
        raise ValueError("Experiment config digest mismatch")
    package = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((package / "train").rglob("*.py"))
    }
    source_hash = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if source_hash != os.environ["DOKU_SOURCE_SHA256"]:
        raise ValueError("Source digest mismatch")
    if (
        hashlib.sha256((package / "uv.lock").read_bytes()).hexdigest()
        != os.environ["DOKU_LOCK_SHA256"]
    ):
        raise ValueError("Lock digest mismatch")
    root = Path(os.environ["DOKU_RUN_ROOT"])
    if root.exists():
        raise FileExistsError(root)
    publish = ArtifactUpload(
        boto3.client("s3"), os.environ["DOKU_ARTIFACT_URI"], os.environ["DOKU_KMS_KEY"]
    )
    try:
        if os.environ.get("DOKU_MODE") == "daily":
            from .daily_train_model import execute as daily_execute

            def monitoring_source(root):
                context = os.environ.get("DOKU_MONITORING_CONTEXT")
                if not context:
                    return
                from recommend_tf.platforms.monitoring.report_metrics import (
                    observation,
                    source_evidence,
                )

                context = json.loads(context)
                status_path = root / "daily-status.json"
                metadata = json.loads(status_path.read_text())
                metadata["training_job_name"] = context["training_job_name"]
                status_path.write_text(json.dumps(metadata, sort_keys=True))
                event = observation(
                    root,
                    context,
                    run_uri=os.environ["DOKU_ARTIFACT_URI"],
                    evidence=source_evidence(root),
                )
                (root / "training/monitoring-event.json").write_text(
                    json.dumps(event, sort_keys=True)
                )

            status = daily_execute(
                json.loads(payload),
                root,
                publish=publish,
                on_completed=monitoring_source,
                cutoff=os.environ["DOKU_CUTOFF"],
                frozen_input=os.environ["DOKU_FROZEN_INPUT"],
            )
        elif os.environ.get("DOKU_MODE", "experiment") == "experiment":
            status = execute(json.loads(payload), root, publish=publish)
        else:
            raise ValueError("Unsupported training mode")
        output = Path("/opt/ml/output/data")
        output.mkdir(parents=True, exist_ok=True)
        (output / "experiment-status.json").write_text(json.dumps(status, indent=2))
        (output / "artifact-location.json").write_text(
            json.dumps({"uri": os.environ["DOKU_ARTIFACT_URI"]})
        )
        print(json.dumps(status), flush=True)
    except Exception as error:
        failure = Path("/opt/ml/output/failure")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text("Experiment failed: " + type(error).__name__)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
