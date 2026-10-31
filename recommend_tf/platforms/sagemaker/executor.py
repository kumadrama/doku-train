"""ECS adapter for the existing publisher; explicit, digest-bound S3 inputs."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .daily_pipeline import execute, validate_binding
from .publish_model import location

MAX_INPUT_BYTES = 8 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("metadata redirect forbidden")


def task_identity(metadata_uri, *, opener=None):
    """Only query the ECS task metadata endpoint, without proxies or redirects."""
    uri = urlsplit(metadata_uri)
    if (
        uri.scheme != "http"
        or uri.hostname != "169.254.170.2"
        or uri.port not in (None, 80)
        or uri.username
        or uri.password
        or uri.query
        or uri.fragment
        or not uri.path.startswith("/v4/")
    ):
        raise ValueError("ECS v4 metadata URI required")
    opener = opener or build_opener(ProxyHandler({}), NoRedirect())
    target = metadata_uri.rstrip("/") + "/task"
    with opener.open(target, timeout=5) as response:
        if response.geturl() != target:
            raise ValueError("metadata redirect forbidden")
        body = response.read(MAX_INPUT_BYTES + 1)
    if len(body) > MAX_INPUT_BYTES:
        raise ValueError("metadata too large")
    record = json.loads(body)
    arn = record["TaskARN"]
    if not isinstance(arn, str) or ":ecs:" not in arn or ":task/" not in arn:
        raise ValueError("invalid ECS TaskARN")
    return arn, record["Cluster"]


def download_json(s3, uri, expected, output):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("explicit input SHA256 required")
    bucket, key = location(uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    with response["Body"] as stream:
        body = stream.read(MAX_INPUT_BYTES + 1)
    if len(body) > MAX_INPUT_BYTES or hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("input size or SHA256 mismatch")
    value = json.loads(body)
    Path(output).write_bytes(body)
    return value


def run(args, *, s3, metadata_uri, invoke=execute, identity=task_identity):
    arn, cluster = identity(metadata_uri)
    with tempfile.TemporaryDirectory(prefix="tf-executor-") as directory:
        root = Path(directory)
        values = {}
        for name in ("config", "ready", "binding"):
            values[name] = download_json(
                s3, getattr(args, name + "_uri"), getattr(args, name + "_sha256"), root / name
            )
        binding = values["binding"]
        validate_binding(binding)
        cluster_name = cluster.rsplit("/", 1)[-1]
        if binding["region"] != args.region or binding["ecs_cluster"] not in {
            cluster,
            cluster_name,
        }:
            raise ValueError("task environment differs from binding")
        if arn.split(":")[3] != args.region:
            raise ValueError("task region differs from binding")
        evidence = None
        if args.bootstrap_evidence_uri:
            if not args.bootstrap_run:
                raise ValueError("bootstrap evidence requires run")
            evidence = str(root / "evidence")
            download_json(s3, args.bootstrap_evidence_uri, args.bootstrap_evidence_sha256, evidence)
        return invoke(
            SimpleNamespace(
                **{name: str(root / name) for name in values},
                executor_arn=arn,
                bootstrap_run=args.bootstrap_run,
                bootstrap_evidence=evidence,
                rollback_version=args.rollback_version,
                retry_id=args.retry_id,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    for name in ("config", "ready", "binding"):
        parser.add_argument("--" + name + "-uri", required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--bootstrap-run")
    parser.add_argument("--bootstrap-evidence-uri")
    parser.add_argument("--bootstrap-evidence-sha256")
    parser.add_argument("--rollback-version")
    parser.add_argument("--retry-id", default="")
    args = parser.parse_args()
    import boto3

    run(
        args,
        s3=boto3.Session(region_name=args.region).client("s3"),
        metadata_uri=os.environ["ECS_CONTAINER_METADATA_URI_V4"],
    )


if __name__ == "__main__":
    main()
