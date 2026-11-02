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


def handle(event, settings, *, ecs, s3, events, policy=None):
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
    if settings.get("release"):
        ready["config_sha256"] = settings["release"]["config_sha256"]
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
    if settings.get("release"):
        from .recover_daily import enqueue_ready

        detail = enqueue_ready(s3, settings, ready, detail)
    if policy is not None and not policy.allowed("training"):
        return {"status": "deferred_training", "ready_uri": detail["ready_uri"]}
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
    from pathlib import Path

    release_path = Path(__file__).with_name("ready_release.json")
    if release_path.exists():
        settings = with_release(settings, json.loads(release_path.read_text()))
    elif settings.get("runtime_policy"):
        raise ValueError("runtime policy requires packaged ready release")
    session = boto3.Session(region_name=settings["region"])
    from botocore.config import Config

    from recommend_tf.platforms.runtime_policy import RuntimePolicy

    policy = (
        RuntimePolicy(
            session.client(
                "appconfigdata",
                config=Config(
                    connect_timeout=3,
                    read_timeout=5,
                    retries={"mode": "standard", "max_attempts": 2},
                ),
            ),
            settings["runtime_policy"],
        )
        if settings.get("runtime_policy")
        else None
    )
    if event.get("source") == "aws.events":
        if event.get("resources") != [settings.get("recovery_rule_arn")]:
            raise ValueError("untrusted recovery schedule")
        from .recover_daily import recover

        if policy is None:
            raise ValueError("recovery requires runtime policy")
        return recover(
            session.client("s3"), session.client("events"), settings, policy, session.client("ecs")
        )
    return handle(
        event,
        settings,
        policy=policy,
        **{name: session.client(name) for name in ("ecs", "s3", "events")},
    )


def with_release(settings, packaged):
    """Window belongs to the application release, never hot runtime policy."""
    if packaged["schema_version"] != "doku-tf-ready-release/v1":
        raise ValueError("unsupported ready release")
    release = packaged["release"]
    config_bytes = packaged["config_json"].encode()
    if sha(config_bytes) != release["config_sha256"]:
        raise ValueError("packaged recipe digest mismatch")
    config = json.loads(config_bytes)
    window = config["online_config"]
    if any(
        type(window[key]) is not int or window[key] < 1
        for key in ("training_date", "validation_date")
    ):
        raise ValueError("invalid recipe window")
    if not settings.get("deferred_uri"):
        raise ValueError("ready release requires durable recovery storage")
    return dict(
        settings,
        release=release,
        training_days=window["training_date"],
        validation_days=window["validation_date"],
    )
