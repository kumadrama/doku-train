"""Explicit CAS repair of an unsubmitted legacy daily training request."""

import argparse
import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path

from .monitoring_context import attach_context, validate_environment
from .publish_model import ReleaseState, describe, location


def repair(s3, sm, ecs, binding, *, operation, image, source_sha256, work, apply=False):
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ValueError("explicit immutable image and source digest required")
    state = ReleaseState(s3, binding["state_uri"], binding["kms_key"])
    active = state.value.get("active")
    if not active or active["operation"] != operation or active["stage"] != "training":
        raise ValueError("expected active training operation required")
    executor = active.get("executor")
    if not executor:
        raise ValueError("previous executor identity required")
    observed = ecs.describe_tasks(cluster=binding["ecs_cluster"], tasks=[executor])
    tasks = observed.get("tasks", [])
    if (
        observed.get("failures")
        or len(tasks) != 1
        or tasks[0].get("taskArn") != executor
        or tasks[0].get("lastStatus") != "STOPPED"
    ):
        raise ValueError("cannot prove previous executor stopped")
    original = active["training_request"]
    job = original["TrainingJobName"]
    if describe(sm, "describe_training_job", TrainingJobName=job) is not None:
        raise ValueError("existing training job must not be modified or retrained")
    inline = original["Environment"].get("DOKU_MONITORING_CONTEXT")
    if not inline or len(inline) <= 512:
        raise ValueError("repair requires an oversized legacy monitoring context")
    if (
        image == original["AlgorithmSpecification"]["TrainingImage"]
        or source_sha256 == original["Environment"]["DOKU_SOURCE_SHA256"]
    ):
        raise ValueError("file transport requires an upgraded training image and source digest")
    if any(
        name in original["Environment"]
        for name in ("DOKU_MONITORING_CONTEXT_PATH", "DOKU_MONITORING_CONTEXT_SHA256")
    ):
        raise ValueError("ambiguous legacy context transport")
    context = json.loads(inline)
    if context.get("training_job_name") != job:
        raise ValueError("monitoring context job identity differs")
    channels = [c for c in original["InputDataConfig"] if c["ChannelName"] == "daily"]
    if len(channels) != 1 or channels[0].get("InputMode") != "File":
        raise ValueError("one frozen daily File channel required")
    source = channels[0]["DataSource"]["S3DataSource"]
    if source["S3DataType"] != "S3Prefix":
        raise ValueError("frozen daily S3Prefix required")
    prefix = source["S3Uri"].rstrip("/")
    expected_prefix = binding["work_prefix"].rstrip("/") + "/" + operation + "/daily"
    if prefix != expected_prefix:
        raise ValueError("daily input does not belong to expected operation")
    # Validate the frozen config before changing any object; never rebuild samples.
    bucket, key = location(prefix + "/config.json")
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as stream:
        config_digest = hashlib.sha256(stream.read()).hexdigest()
    if config_digest != original["Environment"]["DOKU_CONFIG_SHA256"]:
        raise ValueError("frozen config digest differs")
    request = copy.deepcopy(original)
    request["AlgorithmSpecification"]["TrainingImage"] = image
    request["Environment"]["DOKU_SOURCE_SHA256"] = source_sha256
    body = json.dumps(context, sort_keys=True, allow_nan=False).encode()
    env = request["Environment"]
    del env["DOKU_MONITORING_CONTEXT"]
    env["DOKU_MONITORING_CONTEXT_PATH"] = "/opt/ml/input/data/daily/monitoring-context.json"
    env["DOKU_MONITORING_CONTEXT_SHA256"] = hashlib.sha256(body).hexdigest()
    validate_environment(request)
    plan = {
        "operation": operation,
        "training_job_name": job,
        "state_etag": state.etag,
        "context_uri": prefix + "/monitoring-context.json",
        "original_request": original,
        "replacement_request": request,
        "applied": False,
    }
    if apply:
        attach_context(s3, request, context, prefix=prefix, work=work, kms_key=binding["kms_key"])
        # Preserve evidence of the rejected request and all existing release state.
        active["monitoring_context_repair"] = {"original_request": copy.deepcopy(original)}
        active["training_request"] = request
        state.save()
        plan["applied"] = True
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--training-image", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import boto3

    binding = json.loads(Path(args.binding).read_text())
    session = boto3.Session(region_name=binding["region"])
    with tempfile.TemporaryDirectory() as directory:
        result = repair(
            session.client("s3"),
            session.client("sagemaker"),
            session.client("ecs"),
            binding,
            operation=args.operation,
            image=args.training_image,
            source_sha256=args.source_sha256,
            work=Path(directory),
            apply=args.apply,
        )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
