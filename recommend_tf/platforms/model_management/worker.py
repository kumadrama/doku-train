"""Execute one verified managed request in the existing CPU pipeline."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ..sagemaker.executor import task_identity
from .contracts import canonical, validate_group
from .requests import validate_request
from .storage import Documents, Journal


def run(documents, group, pin, *, identity, execute):
    request, values = validate_request(documents, group, pin)
    group = values["group"]
    arn, cluster, family = identity
    worker = group["worker"]
    if (
        cluster.rsplit("/", 1)[-1] != worker["cluster"].rsplit("/", 1)[-1]
        or family != worker["task_definition"].split("/")[-1].rsplit(":", 1)[0]
    ):
        raise ValueError("worker identity differs from managed task group")
    journal = Journal(documents, request["slot_uri"])
    if journal.value["request"] != pin or journal.value["status"] not in {
        "starting",
        "submitted",
        "running",
    }:
        raise ValueError("request was cancelled, replaced, or already completed")
    if journal.value.get("task_arn", arn) != arn:
        raise ValueError("another executor owns this request")
    journal.value.update(status="running", task_arn=arn)
    journal.save()
    with tempfile.TemporaryDirectory(prefix="managed-model-") as folder:
        root = Path(folder)
        for key, value in values.items():
            (root / key).write_bytes(canonical(value))
        args = SimpleNamespace(
            config=str(root / "config"),
            ready=str(root / "ready"),
            binding=str(root / "binding"),
            executor_arn=arn,
            bootstrap_run=request["bootstrap_run"],
            bootstrap_evidence=str(root / "bootstrap_evidence")
            if request["bootstrap_run"]
            else None,
            rollback_version=request["rollback_version"],
            retry_id=request["retry_id"],
        )
        try:
            outcome = execute(args)
        except Exception as error:
            journal.value.update(status="failed", error_type=type(error).__name__)
            journal.save()
            raise
    journal.value.update(status=outcome, error_type=None)
    journal.save()
    return journal.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("region", "group-uri", "group-sha256", "request-uri", "request-sha256"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    import boto3

    from ..sagemaker.daily_pipeline import execute
    from ..sagemaker.executor import download_json

    s3 = boto3.Session(region_name=args.region).client("s3")
    with tempfile.TemporaryDirectory() as folder:
        group = validate_group(
            download_json(s3, args.group_uri, args.group_sha256, Path(folder) / "group")
        )
    if group["region"] != args.region:
        raise ValueError("worker region mismatch")
    result = run(
        Documents(s3, group),
        group,
        {"uri": args.request_uri, "sha256": args.request_sha256},
        identity=task_identity(os.environ["ECS_CONTAINER_METADATA_URI_V4"]),
        execute=execute,
    )
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
