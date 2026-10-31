"""Translate verified upstream ECS completion into an immutable daily ready envelope."""

import hashlib
import json
import os
from datetime import date, timedelta


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    with response["Body"] as stream:
        body = stream.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ValueError("metadata exceeds budget")
    return body, response


def cutoff_from_task(task, settings):
    if (
        task["clusterArn"] != settings["cluster_arn"]
        or task["lastStatus"] != "STOPPED"
        or task["taskDefinitionArn"].rsplit(":", 1)[0] != settings["upstream_definition"]
    ):
        raise ValueError("untrusted upstream task")
    containers = task["containers"]
    if not containers or any(c.get("exitCode") != 0 for c in containers):
        raise ValueError("upstream task did not succeed")
    overrides = task.get("overrides", {}).get("containerOverrides", [])
    commands = [c.get("command", []) for c in overrides if c["name"] == settings["container"]]
    if len(commands) != 1:
        raise ValueError("missing upstream command")
    command = commands[0]
    prefix = [
        "fyp-rank-samples",
        "join",
        "--environment",
        settings["environment"],
        "--mode",
        "daily",
        "--project-training",
        "--date",
    ]
    if command[:-1] != prefix:
        return None
    return date.fromisoformat(command[-1])


def build_ready(s3, settings, cutoff):
    sources = []
    for offset in reversed(range(settings["training_days"] + settings["validation_days"])):
        day = (cutoff - timedelta(days=offset)).isoformat()
        key = settings["daily_prefix"].rstrip("/") + "/date=" + day + "/_SUCCESS"
        body, response = read(s3, settings["bucket"], key)
        pointer = json.loads(body)
        prefix = "s3://" + settings["bucket"] + "/"
        if (
            pointer["pointer_version"] != "doku-train-day/v1"
            or pointer["date"] != day
            or not pointer["manifest_uri"].startswith(prefix + key.rsplit("/", 1)[0] + "/run_id=")
        ):
            raise ValueError("invalid published day pointer")
        manifest_key = pointer["manifest_uri"][len(prefix) :]
        manifest_body, _ = read(s3, settings["bucket"], manifest_key)
        manifest = json.loads(manifest_body)
        if (
            sha(manifest_body) != pointer["manifest_sha256"]
            or len(manifest.get("source_markers", {})) != 24
            or manifest["projection_sha256"] != pointer["projection_sha256"]
        ):
            raise ValueError("published manifest evidence differs")
        contracts = {}
        for name in ("features", "labels"):
            raw, _ = read(
                s3,
                settings["bucket"],
                manifest_key.rsplit("/", 1)[0] + "/contracts/" + name + ".json",
            )
            contracts[name] = sha(raw)
            if manifest.get("contract_sha256", {}).get(name, contracts[name]) != contracts[name]:
                raise ValueError("published contract checksum differs")
        sources.append(
            dict(
                date=day,
                status="published",
                manifest_uri=pointer["manifest_uri"],
                manifest_sha256=pointer["manifest_sha256"],
                contract_sha256=contracts,
                maturity_evidence=dict(
                    contract="doku-train-day/v1",
                    success_uri=prefix + key,
                    success_sha256=sha(body),
                    success_version_id=response.get("VersionId"),
                    success_body=pointer,
                ),
            )
        )
    return dict(
        version="doku-tf-dataset-ready/v1",
        timezone="UTC",
        cutoff=cutoff.isoformat(),
        sources=sources,
    )


def handle(event, settings, *, ecs, s3, events):
    if event.get("source") != "aws.ecs" or event.get("detail-type") != "ECS Task State Change":
        raise ValueError("ECS completion event required")
    result = ecs.describe_tasks(cluster=settings["cluster_arn"], tasks=[event["detail"]["taskArn"]])
    if result.get("failures") or len(result["tasks"]) != 1:
        raise ValueError("upstream task cannot be verified")
    task = result["tasks"][0]
    cutoff = cutoff_from_task(task, settings)
    if cutoff is None:
        return {"status": "ignored", "reason": "not_daily_projection"}
    ready = build_ready(s3, settings, cutoff)
    body = json.dumps(ready, sort_keys=True).encode()
    digest = sha(body)
    key = settings["ready_prefix"].rstrip("/") + "/" + digest + "/ready.json"
    try:
        s3.put_object(
            Bucket=settings["bucket"],
            Key=key,
            Body=body,
            IfNoneMatch="*",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=settings["kms_key"],
            ContentType="application/json",
        )
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] != "PreconditionFailed":
            raise
        existing, _ = read(s3, settings["bucket"], key)
        if existing != body:
            raise ValueError("immutable ready conflict") from error
    detail = dict(
        version="doku-tf-ready-event/v1",
        status="published",
        ready_uri="s3://" + settings["bucket"] + "/" + key,
        ready_sha256=digest,
    )
    response = events.put_events(
        Entries=[
            dict(
                Source=settings["event_source"],
                DetailType=settings["detail_type"],
                EventBusName=settings["event_bus"],
                Detail=json.dumps(detail),
            )
        ]
    )
    if response["FailedEntryCount"]:
        raise RuntimeError("ready event delivery failed")
    print(json.dumps(dict(status="ready", cutoff=cutoff.isoformat(), ready_sha256=digest)))
    return detail


def handler(event, context):
    import boto3

    settings = json.loads(os.environ["SETTINGS"])
    session = boto3.Session(region_name=settings["region"])
    return handle(
        event, settings, **{name: session.client(name) for name in ("ecs", "s3", "events")}
    )
