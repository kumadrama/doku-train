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
        status = execute(json.loads(payload), root, publish=publish)
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
