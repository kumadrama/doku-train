"""Replay frozen deferred execution envelopes through the existing ready rule."""

import hashlib
import json
from pathlib import Path

from .publish_model import ReleaseState, immutable_put, location


def enqueue(s3, binding, args, root):
    fields = (
        "config_uri",
        "config_sha256",
        "binding_uri",
        "binding_sha256",
        "ready_uri",
        "ready_sha256",
        "bootstrap_run",
        "bootstrap_evidence_uri",
        "bootstrap_evidence_sha256",
        "rollback_version",
        "retry_id",
    )
    record = {
        "version": "doku-tf-deferred/v1",
        "arguments": {name: getattr(args, name, None) for name in fields},
    }
    from .daily_pipeline import operation_identity

    root = Path(root)
    config = json.loads((root / "config").read_text())
    ready = json.loads((root / "ready").read_text())
    evidence_path = root / "evidence"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    record["cutoff"] = ready["cutoff"]
    record["operation"] = operation_identity(
        config,
        ready,
        binding,
        bootstrap_run=getattr(args, "bootstrap_run", None),
        evidence=evidence,
        rollback_version=getattr(args, "rollback_version", None),
        retry_id=getattr(args, "retry_id", ""),
    )
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(raw).hexdigest()
    uri = (
        binding["deferred_uri"].rstrip("/")
        + "/requests/"
        + str(99999999 - int(ready["cutoff"].replace("-", "")))
        + "/"
        + sha
        + ".json"
    )
    path = Path(root) / "deferred.json"
    path.write_bytes(raw)
    immutable_put(s3, uri, path, binding["kms_key"])
    return uri, sha


def enqueue_ready(s3, settings, ready, detail):
    """Persist producer intent before checking policy or delivering an event."""
    release = settings["release"]
    record = {
        "version": "doku-tf-deferred/v1",
        "cutoff": ready["cutoff"],
        "operation": None,
        "arguments": {
            **{
                key: release[key]
                for key in ("config_uri", "config_sha256", "binding_uri", "binding_sha256")
            },
            "ready_uri": detail["ready_uri"],
            "ready_sha256": detail["ready_sha256"],
            "bootstrap_run": None,
            "bootstrap_evidence_uri": None,
            "bootstrap_evidence_sha256": None,
            "rollback_version": None,
            "retry_id": "",
        },
    }
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    uri = (
        settings["deferred_uri"].rstrip("/")
        + "/requests/"
        + str(99999999 - int(ready["cutoff"].replace("-", "")))
        + "/"
        + digest
        + ".json"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "request.json"
        path.write_bytes(raw)
        immutable_put(s3, uri, path, settings["kms_key"])
    return dict(detail, ready_uri=uri, ready_sha256=digest)


def set_status(s3, binding, sha, status):
    bucket, key = location(binding["deferred_uri"].rstrip("/") + "/status/" + sha + ".json")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps({"status": status}).encode(),
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=binding["kms_key"],
    )


def recover(s3, events, settings, policy, ecs=None):
    if not policy.allowed("recovery"):
        return {"status": "paused"}
    budget = policy.read()["recovery_max_items"]
    bucket, prefix = location(settings["deferred_uri"].rstrip("/") + "/requests/")
    cursor_key = prefix.rsplit("requests/", 1)[0] + "cursor.json"
    try:
        response = s3.get_object(Bucket=bucket, Key=cursor_key)
        cursor = json.loads(response["Body"].read())
        etag = response["ETag"]
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
            raise
        cursor, etag = {}, None
    page = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=budget,
        **({"ContinuationToken": cursor["token"]} if cursor.get("token") else {}),
    )
    sent = 0
    for item in page.get("Contents", []):
        uri = "s3://" + bucket + "/" + item["Key"]
        response = s3.get_object(Bucket=bucket, Key=item["Key"])
        raw = response["Body"].read(65537)
        if len(raw) > 65536:
            raise ValueError("oversized deferred envelope")
        sha = hashlib.sha256(raw).hexdigest()
        if item["Key"].rsplit("/", 1)[-1] != sha + ".json":
            raise ValueError("deferred envelope digest mismatch")
        record = json.loads(raw)
        if record.get("version") != "doku-tf-deferred/v1":
            raise ValueError("unsupported deferred envelope")
        if record["operation"] is None and not policy.allowed("training"):
            continue
        state = ReleaseState(s3, settings["state_uri"], settings["kms_key"]).value
        active = state["active"]
        prior = next(
            (entry for entry in state["history"] if entry["operation"] == record["operation"]), None
        )
        if prior is not None:
            set_status(s3, settings, sha, prior["stage"])
            continue
        if active:
            if active["operation"] != record["operation"]:
                continue
            if active.get("executor"):
                if ecs is None:
                    raise ValueError("executor liveness client required")
                observed = ecs.describe_tasks(
                    cluster=settings["cluster_arn"], tasks=[active["executor"]]
                )
                if observed.get("failures") or len(observed.get("tasks", [])) != 1:
                    raise ValueError("cannot prove previous executor stopped")
                if observed["tasks"][0]["lastStatus"] != "STOPPED":
                    continue
        elif (
            not record["arguments"].get("rollback_version")
            and state["high_water"]
            and record["cutoff"] <= state["high_water"]
        ):
            set_status(s3, settings, sha, "superseded")
            continue
        status_key = prefix.rsplit("requests/", 1)[0] + "status/" + sha + ".json"
        try:
            status = json.loads(s3.get_object(Bucket=bucket, Key=status_key)["Body"].read())
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") not in {
                "NoSuchKey",
                "404",
            }:
                raise
            # An envelope without a status may be a crash between writes; replay is safe
            # because the original journal/executor lease owns the operation.
            status = {"status": "deferred_unknown"}
        if not status["status"].startswith("deferred_"):
            continue
        result = events.put_events(
            Entries=[
                dict(
                    Source=settings["event_source"],
                    DetailType=settings["detail_type"],
                    EventBusName=settings["event_bus"],
                    Detail=json.dumps(
                        dict(
                            version="doku-tf-ready-event/v1",
                            status="published",
                            ready_uri=uri,
                            ready_sha256=sha,
                        )
                    ),
                )
            ]
        )
        if result["FailedEntryCount"]:
            raise RuntimeError("deferred delivery failed")
        sent += 1
    s3.put_object(
        Bucket=bucket,
        Key=cursor_key,
        Body=json.dumps({"token": page.get("NextContinuationToken")}).encode(),
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=settings["kms_key"],
        **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
    )
    return {"status": "replayed", "count": sent}
