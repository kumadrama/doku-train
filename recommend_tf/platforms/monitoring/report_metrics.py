"""Project sealed selected-epoch evidence into immutable monitoring events."""

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from recommend_tf.train.comm.artifact_integrity import checksum, verify_run
from recommend_tf.train.comm.epoch_metrics import read_selected_metrics
from recommend_tf.train.comm.model_identity import model_name, recipe_sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def observation(
    root, binding, *, run_uri, training_job=None, historical_identity=None, evidence=None
):
    root = Path(root)
    evidence = verify_run(root, evidence)
    required = {"experiment-config.json", "daily-status.json", "training/selection.json"}
    if not required <= evidence["files"].keys():
        raise ValueError("sealed daily run evidence required")
    status = json.loads((root / "daily-status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("selected successful daily run required")
    config = json.loads((root / "experiment-config.json").read_text())
    name = model_name(config)
    identity_source = None
    if name is None:
        if not historical_identity or historical_identity["config_sha256"] != checksum(
            root / "experiment-config.json"
        ):
            raise ValueError("historical model identity requires the original config digest")
        name = model_name(historical_identity, required=True)
        identity_source = historical_identity
    if name != binding["model_name"]:
        raise ValueError("configuration and deployment model_name differ")
    summary = read_selected_metrics(root, config)
    epoch = summary["selection"]["epoch"]
    sources = {
        "training/selection.json",
        f"training/epoch-metrics/{epoch:04d}.json",
        f"training/epoch-{epoch:04d}/checkpoint.json",
        "input/manifest.json",
    }
    if not sources <= evidence["files"].keys():
        raise ValueError("selected metric sources must be sealed")
    completed = status.get("training_completed_at_ms")
    if completed is None:
        if not training_job or training_job["TrainingJobStatus"] != "Completed":
            raise ValueError("original training completion evidence required")
        if training_job.get("Environment", {}).get("DOKU_ARTIFACT_URI", "").rstrip(
            "/"
        ) != run_uri.rstrip("/"):
            raise ValueError("training completion job does not own this run URI")
        completed = int(training_job["TrainingEndTime"].timestamp() * 1000)
    if type(completed) is not int or completed <= 0:
        raise ValueError("invalid original completion time")
    tasks = []
    mapping = binding.get("monitoring", {}).get("metric_mapping", {})
    if not set(mapping) <= binding["outputs"].keys():
        raise ValueError("display mapping contains an unknown primary task")
    for task, unit in sorted(binding["outputs"].items()):
        kind = {"probability": "auc", "seconds": "mse"}.get(unit)
        metrics = summary["metrics"][task]
        candidates = [name for name, metric in metrics.items() if metric["func"] == kind]
        metric = mapping.get(task)
        if metric is None:
            if len(candidates) != 1:
                raise ValueError("ambiguous display metric: " + task)
            metric = candidates[0]
        result = metrics[metric]
        if result["func"] not in {"auc", "mse"} or (kind and result["func"] != kind):
            raise ValueError("display metric does not match output unit")
        tasks.append(
            {
                "task": task,
                "metric_name": metric,
                "metric_kind": result["func"],
                "value": result["value"],
                "reason": result.get("reason"),
                "valid_samples": result.get("rows"),
                "unit": unit,
            }
        )
    cutoff = status["cutoff"]
    day = date.fromisoformat(cutoff)
    day_start = int(datetime.combine(day, time(), timezone(timedelta(hours=8))).timestamp() * 1000)
    event = {
        "schema_version": "doku-selected-model-metrics/v1",
        "event_type": "training_selected",
        "environment": binding["environment"],
        "model_name": name,
        "model_type": config.get("model_type", "rerank"),
        "recipe_sha256": recipe_sha256(config),
        "run_uri": run_uri,
        "run_id": digest({"run_uri": run_uri}),
        "source_sha256": digest(
            {name: evidence["files"][name] for name in sorted(required | sources)}
        ),
        "manifest_sha256": evidence["files"]["input/manifest.json"],
        "data_cutoff": cutoff,
        "day_start_ms": day_start,
        "dataset_timezone": "UTC",
        "evaluated_dates": config["dataset_config"]["split"]["validation_dates"],
        "training_completed_at_ms": completed,
        "selected_epoch": epoch,
        "checkpoint_identity": summary["checkpoint_identity"],
        "selection_score": summary["selection"].get("score"),
        "evaluation_split": summary["evaluation_split"],
        "display_label": "Test",
        "tasks": tasks,
        "deployment": {
            "region": binding["region"],
            "endpoint_name": binding["deployment"]["endpoint_name"],
        },
    }
    if status.get("training_job_name"):
        event["training_job_name"] = status["training_job_name"]
    if training_job and "training_completed_at_ms" not in status:
        event["training_job_name"] = training_job["TrainingJobName"]
        event["training_job_arn"] = training_job.get("TrainingJobArn")
    if identity_source:
        event["historical_identity"] = identity_source
    event["event_id"] = digest(
        {
            k: event[k]
            for k in (
                "schema_version",
                "environment",
                "model_name",
                "run_uri",
                "selected_epoch",
                "evaluation_split",
            )
        }
    )
    return event


def put_event(s3, event, options):
    """One immutable key per logical event. Retry verifies bytes; conflicts never overwrite."""
    from recommend_tf.platforms.sagemaker.publish_model import location, read

    uri = options["outbox_uri"].rstrip("/") + "/" + event["event_id"] + ".json"
    bucket, key = location(uri)
    body = canonical(event)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            IfNoneMatch="*",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=options["kms_key"],
        )
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code not in {"PreconditionFailed", "412"}:
            raise
        previous, _ = read(s3, uri)
        if previous != body:
            raise ValueError(
                "monitoring event payload conflicts with immutable identity"
            ) from error
    return uri


def report(s3, root, binding, **kwargs):
    return put_event(s3, observation(root, binding, **kwargs), binding["monitoring"])


def release_event(state_uri, record, binding):
    """Called only after the terminal journal record is durable."""
    name = model_name(binding, required=True)
    stage = record["stage"]
    status = {"deployed": "published", "rolled_back": "rolled_back"}.get(stage, "failed")
    target = record.get("rollback_target", {}) if stage == "rolled_back" else record
    event = {
        "schema_version": "doku-model-release-event/v1",
        "event_type": "model_release",
        "environment": binding["environment"],
        "model_name": name,
        "state_uri": state_uri,
        "operation": record["operation"],
        "release_revision": record["release_revision"],
        "occurred_at_ms": record["finished_at_ms"],
        "status": status,
        "stage": stage,
        "run_uri": record.get("run_uri"),
        "model_version": target.get("model_version"),
        "endpoint_config": target.get("endpoint_config"),
        "deployment": {
            "region": binding["region"],
            "endpoint_name": binding["deployment"]["endpoint_name"],
        },
    }
    if status != "failed" and not event["model_version"]:
        raise ValueError("successful release requires a confirmed target version")
    event["event_id"] = digest(
        {k: event[k] for k in ("schema_version", "state_uri", "operation", "release_revision")}
    )
    return event


def report_release(s3, state, binding):
    for record in state.value["history"]:
        # Old journals require explicit provenance backfill, not invented timestamps.
        if "release_revision" in record:
            put_event(s3, release_event(state.uri, record, binding), binding["monitoring"])


def source_evidence(root):
    """Small pre-seal inventory for the local completion callback, excluding prepared data."""
    root = Path(root)
    selection = json.loads((root / "training/selection.json").read_text())
    epoch = selection["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("invalid selected epoch")
    names = {
        "experiment-config.json",
        "daily-status.json",
        "training/selection.json",
        "input/manifest.json",
        f"training/epoch-metrics/{epoch:04d}.json",
        f"training/epoch-{epoch:04d}/checkpoint.json",
    }
    return {"version": "doku-tf-run/v1", "files": {name: checksum(root / name) for name in names}}
