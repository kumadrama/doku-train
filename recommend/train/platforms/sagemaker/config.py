from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    environment: str
    region: str
    account_id: str
    role_arn: str
    subnet_ids: list[str] = Field(min_length=1)
    security_group_ids: list[str] = Field(min_length=1)
    input_prefixes: list[str] = Field(min_length=1)
    experiment_uri: str
    kms_key_arn: str

    @model_validator(mode="after")
    def validate_identity(self) -> Binding:
        if self.version != "doku-sagemaker-binding/v1":
            raise ValueError("unsupported binding")
        if not re.fullmatch(r"\d{12}", self.account_id):
            raise ValueError("invalid account")
        if not self.role_arn.startswith(f"arn:aws:iam::{self.account_id}:role/"):
            raise ValueError("role account mismatch")
        if not self.kms_key_arn.startswith(f"arn:aws:kms:{self.region}:{self.account_id}:key/"):
            raise ValueError("KMS identity mismatch")
        for uri in [self.experiment_uri, *self.input_prefixes]:
            split_s3(uri)
            if not uri.endswith("/"):
                raise ValueError("prefix must end in slash")
        return self


class Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    environment: str
    binding: str
    image_uri: str
    source_revision: str
    source_sha256: str
    lock_sha256: str
    instance_type: str = "ml.m5.2xlarge"
    instance_count: int = 1
    volume_size_gb: int = Field(default=100, ge=1)
    max_runtime_seconds: int = Field(default=86400, ge=1)
    loss_interval_steps: int = Field(default=100, ge=1)
    sync_interval_seconds: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def validate_runtime(self) -> Runtime:
        if self.version != "doku-sagemaker-runtime/v1":
            raise ValueError("unsupported runtime")
        if self.instance_type != "ml.m5.2xlarge" or self.instance_count != 1:
            raise ValueError("only single ml.m5.2xlarge supported")
        for value, length in [
            (self.source_revision, 40),
            (self.source_sha256, 64),
            (self.lock_sha256, 64),
        ]:
            if not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
                raise ValueError("invalid image source identity")
        if not re.fullmatch(
            r"\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9/_.-]+@sha256:[0-9a-f]{64}",
            self.image_uri,
        ):
            raise ValueError("immutable ECR image digest required")
        return self


def split_s3(uri: str) -> tuple[str, str]:
    p = urlsplit(uri)
    if p.scheme != "s3" or not p.netloc or not p.path.lstrip("/") or p.query or p.fragment:
        raise ValueError("explicit S3 object/prefix required")
    if any(x in {".", "..", ""} for x in p.path.lstrip("/").rstrip("/").split("/")):
        raise ValueError("invalid S3 path")
    return p.netloc, p.path.lstrip("/")


def allowed(uri: str, prefixes: list[str]) -> None:
    split_s3(uri)
    if not any(uri.startswith(prefix) for prefix in prefixes):
        raise ValueError(f"input outside approved prefixes: {uri}")


def load_runtime(path: Path) -> tuple[Runtime, Binding]:
    runtime = Runtime.model_validate(yaml.safe_load(path.read_text()))
    binding = Binding.model_validate(yaml.safe_load((path.parent / runtime.binding).read_text()))
    if runtime.environment != binding.environment:
        raise ValueError("environment mismatch")
    expected = f"{binding.account_id}.dkr.ecr.{binding.region}.amazonaws.com/"
    if not runtime.image_uri.startswith(expected):
        raise ValueError("image account/region mismatch")
    return runtime, binding


def validate_experiment(config: dict[str, Any], runtime: Runtime) -> None:
    if config["model_type"] != "rerank":
        raise ValueError("initial platform supports DNN/DeepFM only")
    execution = config["execution"]
    if execution["memory_limit_bytes"] > 30 * 1024**3:
        raise ValueError("memory budget must leave at least 2 GiB for system")
    if execution["cache_limit_bytes"] >= runtime.volume_size_gb * 1024**3:
        raise ValueError("cache budget must leave disk for output and temporary files")
