"""Shared scheduler; durable launch intent plus ECS client-token idempotency."""

import time

from ..sagemaker.publish_model import ReleaseState, missing
from .contracts import digest
from .requests import enqueue, next_cutoff, validate_request
from .storage import Journal, conflict


def launch(documents, group, ecs, slot_uri, *, now=time.time):
    journal = Journal(documents, slot_uri)
    record = journal.value
    if record["status"] not in {"queued", "starting"}:
        return record
    request, values = validate_request(documents, group, record["request"])
    group = values["group"]
    if request["slot_uri"] != slot_uri:
        raise ValueError("request slot mismatch")
    state = ReleaseState(documents.s3, values["binding"]["state_uri"], group["kms_key"]).value
    if state["active"]:
        from ..sagemaker.operation_identity import operation_identity

        operation = operation_identity(
            values["config"],
            values["ready"],
            values["binding"],
            bootstrap_run=request["bootstrap_run"],
            evidence=values.get("bootstrap_evidence"),
            rollback_version=request["rollback_version"],
            retry_id=request["retry_id"],
        )
        if not record.get("launch_attempt") or state["active"]["operation"] != operation:
            return {**record, "waiting": "model_release_in_progress"}
    if (
        not request["rollback_version"]
        and state["high_water"]
        and request["cutoff"] <= state["high_water"]
    ):
        record["status"] = "superseded"
        journal.save()
        return record
    gate = Journal(documents, documents.uri("launches/" + request["model_name"] + ".json"))
    if (
        gate.value
        and gate.value["request"] != record["request"]
        and gate.value["slot_uri"] != slot_uri
    ):
        other, _ = documents.read(gate.value["slot_uri"])
        if not other or other["status"] not in {
            "deployed",
            "rolled_back",
            "cancelled",
            "superseded",
        }:
            return {**record, "waiting": "previous_model_request_not_finished"}
    if not gate.value or gate.value["request"] != record["request"]:
        gate.value = {"request": record["request"], "slot_uri": slot_uri}
        gate.save()
    worker = group["worker"]
    token = digest(
        {"request": record["request"], "worker": worker, "attempt": record.get("launch_attempt", 0)}
    )
    started_by = "mm-" + token[:32]
    if record["status"] == "queued":
        record.update(status="starting", started_at=now(), launch_token=token)
        journal.save()  # Cancellation races this CAS; no task exists before it succeeds.
    elif record["launch_token"] != token:
        raise ValueError("worker binding changed during pending launch")
    if now() - record["started_at"] > 3600:
        # ECS idempotency tokens expire: never retry an ambiguous old submission blindly.
        arns = ecs.list_tasks(cluster=worker["cluster"], startedBy=started_by)["taskArns"]
        if len(arns) != 1:
            return {**record, "waiting": "submission_unknown_requires_reconciliation"}
        record.update(status="submitted", task_arn=arns[0])
        journal.save()
        return record
    group_pin = request["runtime"]
    response = ecs.run_task(
        cluster=worker["cluster"],
        taskDefinition=worker["task_definition"],
        launchType="FARGATE",
        count=1,
        clientToken=token,
        startedBy=started_by,
        networkConfiguration=worker["network_configuration"],
        overrides={
            "containerOverrides": [
                {
                    "name": worker["container"],
                    "command": [
                        "python",
                        "-m",
                        "recommend_tf.platforms.model_management.worker",
                        "--region",
                        group["region"],
                        "--group-uri",
                        group_pin["uri"],
                        "--group-sha256",
                        group_pin["sha256"],
                        "--request-uri",
                        record["request"]["uri"],
                        "--request-sha256",
                        record["request"]["sha256"],
                    ],
                }
            ]
        },
    )
    if response.get("failures") or len(response.get("tasks", [])) != 1:
        # A partial/uncertain response keeps the original token for reconciliation.
        raise RuntimeError("managed ECS launch was not acknowledged")
    record.update(status="submitted", task_arn=response["tasks"][0]["taskArn"])
    try:
        journal.save()
    except Exception as error:
        if not conflict(error):
            raise
        current, _ = documents.read(slot_uri)
        if current["request"] != record["request"] or current.get("task_arn") != record["task_arn"]:
            raise
        return current  # Worker may have claimed and completed before run_task returned.
    return record


def tick(documents, group, registry, ecs):
    results = {}
    for name, model in sorted(registry["models"].items()):
        if model["admission"] != "managed":
            continue
        if "adoption" in model:
            owner, _ = documents.read(documents.uri("ownership/" + name + ".json"))
            if not owner or owner["status"] != "managed":
                continue
        try:
            # Existing requests keep their original recipe even after daily is disabled.
            pending = Journal(documents, documents.uri("pending/" + name + ".json"))
            for uri in list((pending.value or {}).get("slots", [])):
                record, _ = documents.read(uri)
                if not record:
                    continue  # Admission intent was recorded before the slot write.
                if record["status"] in {"deployed", "rolled_back", "cancelled", "superseded"}:
                    pending.value["slots"].remove(uri)
                    pending.save()
                    continue
                if record["status"] == "failed" or record["status"].endswith("_failed"):
                    raise ValueError("failed request requires explicit recovery or rebuild")
                if record["status"] in {"deferred_training", "deferred_publication"}:
                    recover(documents, group, ecs, uri, record["request"])
                    continue
                if record.get("task_arn") and record["status"] in {
                    "submitted",
                    "running",
                }:
                    _, frozen = validate_request(documents, group, record["request"])
                    tasks = ecs.describe_tasks(
                        cluster=frozen["group"]["worker"]["cluster"], tasks=[record["task_arn"]]
                    )
                    if tasks.get("failures") or len(tasks.get("tasks", [])) != 1:
                        raise ValueError(
                            "executor status unknown; explicit reconciliation required"
                        )
                    if tasks["tasks"][0]["lastStatus"] == "STOPPED":
                        recover(documents, group, ecs, uri, record["request"])
                else:
                    launch(documents, group, ecs, uri)
            if not model["daily_enabled"]:
                continue
            from .contracts import binding_for

            state = ReleaseState(
                documents.s3, binding_for(group, name, model)["state_uri"], group["kms_key"]
            ).value
            cutoff = next_cutoff(state, model)
            if cutoff:
                record = enqueue(documents, group, registry, name, cutoff=cutoff)
                request = documents.pinned(record["request"])
                results[name] = launch(documents, group, ecs, request["slot_uri"])["status"]
        except Exception as error:
            results[name] = (
                "waiting_for_data" if missing(error) else "failed:" + type(error).__name__
            )
    failures = [name for name, status in results.items() if status.startswith("failed:")]
    if failures:
        raise RuntimeError("managed dispatch failed for: " + ",".join(failures))
    return results


def recover(documents, group, ecs, slot_uri, expected_request):
    """Resume the same frozen request only after proving its previous ECS worker stopped."""
    journal = Journal(documents, slot_uri)
    record = journal.value
    if record["request"] != expected_request or not record.get("task_arn"):
        raise ValueError("owned executor evidence required for recovery")
    if record["status"] in {"deployed", "rolled_back", "cancelled", "superseded"}:
        raise ValueError("terminal request cannot be recovered")
    request, values = validate_request(documents, group, record["request"])
    group = values["group"]
    state = ReleaseState(documents.s3, values["binding"]["state_uri"], group["kms_key"]).value
    from ..sagemaker.operation_identity import operation_identity

    operation = operation_identity(
        values["config"],
        values["ready"],
        values["binding"],
        bootstrap_run=request["bootstrap_run"],
        evidence=values.get("bootstrap_evidence"),
        rollback_version=request["rollback_version"],
        retry_id=request["retry_id"],
    )
    active = state["active"]
    if active and active["operation"] != operation:
        raise ValueError("another release owns this model")
    # The worker writes deferred only after the pipeline released its executor and
    # confirmed no in-flight side effect. This survives ECS STOPPED metadata expiry.
    gracefully_released = record["status"].startswith("deferred_") and (
        not active or not active.get("executor")
    )
    if not gracefully_released:
        response = ecs.describe_tasks(
            cluster=group["worker"]["cluster"], tasks=[record["task_arn"]]
        )
        if (
            response.get("failures")
            or len(response.get("tasks", [])) != 1
            or response["tasks"][0]["lastStatus"] != "STOPPED"
        ):
            raise ValueError("cannot prove previous executor stopped")
    if not active:
        prior = next((r for r in state["history"] if r["operation"] == operation), None)
        if prior:
            successful = prior["stage"] == "deployed" or (
                request["rollback_version"] and prior["stage"] == "rolled_back"
            )
            record["status"] = prior["stage"] if successful else "failed"
            record["release_stage"] = prior["stage"]
            journal.save()
            return record
    record.update(status="queued", launch_attempt=record.get("launch_attempt", 0) + 1)
    record.pop("task_arn")
    journal.save()
    return launch(documents, group, ecs, slot_uri)
