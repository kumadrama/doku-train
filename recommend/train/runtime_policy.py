"""Fresh operational policy at cloud task boundaries; local experiments stay offline."""

from __future__ import annotations

import json
import os
from typing import Any

PROFILE = "doku-train/tree"


class OperationPaused(RuntimeError):
    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"{operation} disabled")


def read_policy(*, client: Any = None) -> dict[str, Any]:
    environment = os.environ.get("DOKU_ENV")
    if environment not in {"dte", "prod"}:
        raise ValueError("AppConfig requires explicit DOKU_ENV=dte/prod")
    try:
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "appconfigdata",
                region_name=os.environ.get("AWS_REGION"),
                config=Config(
                    connect_timeout=3,
                    read_timeout=5,
                    retries={"total_max_attempts": 2, "mode": "standard"},
                ),
            )
        session = client.start_configuration_session(
            ApplicationIdentifier="doku",
            EnvironmentIdentifier=environment,
            ConfigurationProfileIdentifier=PROFILE,
        )
        response = client.get_latest_configuration(
            ConfigurationToken=session["InitialConfigurationToken"]
        )
        stream = response["Configuration"]
        try:
            data = json.loads(stream.read())
        finally:
            stream.close()
        for key in ("training", "publication", "cleanup", "redis"):
            if type(data[key]["enabled"]) is not bool:
                raise ValueError("invalid enabled")
        redis = data["redis"]
        for key in ("timeout_seconds", "max_file_bytes", "max_package_bytes", "max_model_bytes"):
            if type(redis[key]) is not int or redis[key] <= 0:
                raise ValueError("invalid Redis limit")
        if not (redis["max_file_bytes"] <= redis["max_package_bytes"] <= redis["max_model_bytes"]):
            raise ValueError("invalid Redis budgets")
        return dict(data)
    except Exception:
        raise ValueError("APPCONFIG_READ_OR_VALIDATION_FAILED") from None


def policy_for(root: str) -> dict[str, Any] | None:
    return read_policy() if root.startswith("s3://") else None


def check(root: str, operation: str) -> None:
    policy = policy_for(root)
    if policy is None:
        return
    if not policy[operation]["enabled"]:
        raise OperationPaused(operation)
    if operation == "publication" and not policy["redis"]["enabled"]:
        raise OperationPaused("redis")


def skipped(error: OperationPaused) -> dict[str, Any]:
    return {"status": "skipped", "reason": f"{error.operation}_disabled"}
