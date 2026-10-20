from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.platforms.sagemaker.config import (
    Binding,
    Runtime,
    allowed,
    split_s3,
    validate_experiment,
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def put_once(s3: Any, uri: str, body: bytes, key: str) -> None:
    bucket, name = split_s3(uri)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=name,
            Body=body,
            IfNoneMatch="*",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=key,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
            raise
        existing = s3.get_object(Bucket=bucket, Key=name)["Body"].read()
        if existing != body:
            raise ValueError("immutable submission collision") from exc


def metrics(config: dict[str, Any]) -> list[dict[str, str]]:
    names = []
    for task, declaration in config["tasks"].items():
        for split in ["train", "validation", "test"]:
            names.append(f"{split}/{task}/loss")
            if split != "train":
                names += [
                    f"{split}/{task}/{name}"
                    for name, m in declaration["metrics"].items()
                    if m["func"] in {"auc", "regression_auc"}
                ]
    if len(names) > 40:
        raise ValueError("SageMaker supports at most 40 metric definitions; reduce displayed tasks")
    return [
        {"Name": name, "Regex": re.escape(f"SM_METRIC {name}=") + r"([-+0-9.eE]+)"}
        for name in names
    ]


def prepare(
    config: dict[str, Any],
    runtime: Runtime,
    binding: Binding,
    run_id: str,
    s3: Any,
    destination: Path,
    *,
    original_yaml: str | None = None,
) -> dict[str, Any]:
    """Read and freeze metadata only. Never upload or call CreateTrainingJob."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,45}", run_id):
        raise ValueError("run ID must be 1..46 lowercase letters, digits or hyphens")
    config = validate_config(copy.deepcopy(config))
    validate_experiment(config, runtime)
    uri = config["dataset"]["manifest_uri"]
    allowed(uri, binding.input_prefixes)
    parent = uri.rsplit("/", 1)[0] + "/"
    raw = {}
    for name, source in [
        ("manifest.json", uri),
        ("contracts/features.json", parent + "contracts/features.json"),
        ("contracts/labels.json", parent + "contracts/labels.json"),
    ]:
        allowed(source, binding.input_prefixes)
        bucket, key = split_s3(source)
        raw[name] = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    manifest = json.loads(raw["manifest.json"])
    for partition in manifest["partitions"]:
        allowed(partition["uri"], binding.input_prefixes)
    record = {
        "manifest_uri": uri,
        "manifest_file_sha256": hashlib.sha256(raw["manifest.json"]).hexdigest(),
        "manifest_sha256": manifest["content_sha256"],
        "contract_digests": {
            n: hashlib.sha256(raw[f"contracts/{n}.json"]).hexdigest()
            for n in ["features", "labels"]
        },
    }
    raw["record.json"] = canonical(record)
    # Validate using the same reader as training, with a temporary cache (no data reads).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, body in raw.items():
            target = root / "input" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        check = copy.deepcopy(config)
        check["execution"]["cache_dir"] = str(root / "cache")
        PublishedDataset(check, frozen_input=root / "input")
    package = {
        "version": "doku-sagemaker-submission/v1",
        "run_id": run_id,
        "config": config,
        "original_yaml": original_yaml,
        "runtime": runtime.model_dump(),
        "binding": binding.model_dump(),
        "input": {name: body.decode() for name, body in raw.items()},
    }
    payload = canonical(package)
    sha = hashlib.sha256(payload).hexdigest()
    job = f"doku-exp-{run_id}"
    prefix = binding.experiment_uri
    request = {
        "TrainingJobName": job,
        "RoleArn": binding.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": runtime.image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": metrics(config),
        },
        "InputDataConfig": [
            {
                "ChannelName": "submission",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": prefix + f"submissions/{run_id}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": prefix + f"runs/{run_id}/artifacts/",
            "KmsKeyId": binding.kms_key_arn,
        },
        "ResourceConfig": {
            "InstanceType": runtime.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": runtime.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": runtime.max_runtime_seconds},
        "VpcConfig": {
            "Subnets": binding.subnet_ids,
            "SecurityGroupIds": binding.security_group_ids,
        },
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": False,
        "Environment": {"DOKU_SUBMISSION_SHA256": sha},
        "Tags": [
            {"Key": "Project", "Value": "doku-train"},
            {"Key": "Environment", "Value": binding.environment},
            {"Key": "RunId", "Value": run_id},
        ],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "package.json").write_bytes(payload)
    (destination / "request.json").write_bytes(canonical(request))
    return request


def submit(prepared: Path, s3: Any, sm: Any) -> dict[str, Any]:
    payload = (prepared / "package.json").read_bytes()
    package = json.loads(payload)
    request = json.loads((prepared / "request.json").read_bytes())
    sha = hashlib.sha256(payload).hexdigest()
    if request["Environment"]["DOKU_SUBMISSION_SHA256"] != sha:
        raise ValueError("submission checksum mismatch")
    binding = Binding.model_validate(package["binding"])
    uri = binding.experiment_uri + f"submissions/{package['run_id']}/package.json"
    put_once(s3, uri, payload, binding.kms_key_arn)
    name = request["TrainingJobName"]
    try:
        existing = sm.describe_training_job(TrainingJobName=name)
    except ClientError as exc:
        message = exc.response["Error"].get("Message", "").lower()
        if exc.response["Error"]["Code"] != "ValidationException" or not any(
            text in message for text in ("not found", "could not find")
        ):
            raise
    else:
        if existing.get("Environment", {}).get("DOKU_SUBMISSION_SHA256") != sha:
            raise ValueError("job name already belongs to another submission")
        return {"TrainingJobArn": existing["TrainingJobArn"], "reused": True}
    # A lost response is recovered by rerunning submit on the SAME prepared directory.
    return dict(sm.create_training_job(**request))
