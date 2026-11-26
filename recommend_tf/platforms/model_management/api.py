"""Versioned management API for the shared group controller Lambda."""

import json
import os
import tempfile
from pathlib import Path

from ..sagemaker.publish_model import ReleaseState
from . import adoption
from .contracts import API, binding_for, digest, exact, under, validate_group
from .dispatch import launch, tick
from .registry import Registry
from .requests import cancel, enqueue, rebuild
from .storage import Documents


def handle(event, group, *, s3, control, data, ecs, events, sm, resource_client=None):
    documents = Documents(s3, group)
    registry_store = Registry(documents, group, control, data)
    if event.get("source") == "aws.events":
        return tick(documents, group, registry_store.read(), ecs)
    exact(event, ("schema_version", "operation", "arguments"))
    if event["schema_version"] != API:
        raise ValueError("unknown management API")
    op, args = event["operation"], event["arguments"]
    if op == "status":
        exact(args, (), ("model_name",))
        registry = registry_store.read()
        result = {"registry_sha256": digest(registry), "registry": registry}
        if name := args.get("model_name"):
            model = registry["models"][name]
            result["release"] = ReleaseState(
                s3, binding_for(group, name, model)["state_uri"], group["kms_key"]
            ).value
            result["ownership"] = documents.read(documents.uri("ownership/" + name + ".json"))[0]
            result["service"] = binding_for(group, name, model)["deployment"]
            result["requests"] = [
                {**documents.read(u)[0], "slot_uri": u}
                for u in documents.list("slots/" + name + "/")
            ]
        return result
    if op == "plan":
        exact(args, ("action", "model_name", "payload", "expected_registry_sha256"))
        return registry_store.plan(
            args["action"], args["model_name"], args["payload"], args["expected_registry_sha256"]
        )
    if op == "apply":
        exact(args, ("plan",))
        under(args["plan"]["uri"], [documents.uri("plans")])
        intent = documents.pinned(args["plan"])
        if intent["action"] == "adopt":
            return adoption.apply(
                documents, group, registry_store, args["plan"], events=events, ecs=ecs, sm=sm
            )
        if intent["action"] == "register":
            if not group.get("resource_manager") or resource_client is None:
                raise ValueError("group resource enrollment is not configured")
            name = intent["model_name"]
            endpoint = binding_for(group, name, intent["after"]["models"][name])["deployment"][
                "endpoint_name"
            ]
            enrolled = resource_client.invoke(
                FunctionName=group["resource_manager"],
                InvocationType="RequestResponse",
                Payload=json.dumps(
                    {"schema_version": "doku-model-resources/v1", "endpoint_name": endpoint}
                ).encode(),
            )
            if enrolled.get("FunctionError") or enrolled.get("StatusCode") != 200:
                raise RuntimeError("Endpoint resource enrollment failed")
            receipt = json.loads(enrolled["Payload"].read())
            expected_alarms = binding_for(group, name, intent["after"]["models"][name])[
                "deployment"
            ]["alarm_names"]
            if receipt.get("endpoint_name") != endpoint or set(
                receipt.get("alarm_names", [])
            ) != set(expected_alarms):
                raise ValueError("resource enrollment identity or alarms differ")
            documents.immutable("resource-enrollments", receipt)
        return registry_store.apply(args["plan"])
    if op == "adopt_plan":
        exact(args, ("model_name", "legacy_rule", "expected_registry_sha256"))
        registry = registry_store.read()
        if digest(registry) != args["expected_registry_sha256"]:
            raise ValueError("stale registry")
        return adoption.plan(
            documents,
            group,
            registry,
            args["model_name"],
            args["legacy_rule"],
            events=events,
            ecs=ecs,
            sm=sm,
        )
    if op == "adopt_fence":
        exact(args, ("plan",))
        under(args["plan"]["uri"], [documents.uri("plans")])
        return adoption.fence(documents, group, args["plan"], events=events, ecs=ecs)
    if op == "adopt_abort":
        exact(args, ("plan",))
        under(args["plan"]["uri"], [documents.uri("plans")])
        return adoption.abort(documents, group, args["plan"], events=events)
    if op == "submit":
        exact(
            args,
            ("model_name", "cutoff"),
            ("bootstrap_run", "rollback_version", "operation_id", "retry_id"),
        )
        record = enqueue(
            documents,
            group,
            registry_store.read(),
            args["model_name"],
            **{k: v for k, v in args.items() if k != "model_name"},
        )
        request = documents.pinned(record["request"])
        return launch(documents, group, ecs, request["slot_uri"])
    if op in {"cancel", "rebuild", "recover"}:
        exact(
            args,
            ("slot_uri", "expected_request", "retry_id")
            if op == "rebuild"
            else ("slot_uri", "expected_request"),
        )
        under(args["slot_uri"], [documents.uri("slots")])
        if op == "cancel":
            return cancel(documents, args["slot_uri"], args["expected_request"])
        if op == "rebuild":
            return rebuild(
                documents,
                group,
                registry_store.read(),
                args["slot_uri"],
                args["expected_request"],
                retry_id=args["retry_id"],
            )
        from .dispatch import recover

        return recover(documents, group, ecs, args["slot_uri"], args["expected_request"])
    if op == "dispatch":
        exact(args, ())
        return tick(documents, group, registry_store.read(), ecs)
    raise ValueError("unsupported management operation")


def handler(event, context):
    import boto3

    from ..sagemaker.executor import download_json

    # AWS_REGION is supplied by Lambda; resources are injected by platform bindings.
    session = boto3.Session(region_name=os.environ["AWS_REGION"])
    s3 = session.client("s3")
    with tempfile.TemporaryDirectory() as directory:
        group = validate_group(
            download_json(
                s3,
                os.environ["DOKU_MODEL_GROUP_URI"],
                os.environ["DOKU_MODEL_GROUP_SHA256"],
                Path(directory) / "group",
            )
        )
    if group["region"] != session.region_name:
        raise ValueError("controller region differs from group binding")
    result = handle(
        event,
        group,
        s3=s3,
        control=session.client("appconfig"),
        data=session.client("appconfigdata"),
        ecs=session.client("ecs"),
        events=session.client("events"),
        sm=session.client("sagemaker"),
        resource_client=session.client("lambda"),
    )
    print(
        json.dumps(
            {
                "event": "model_management",
                "operation": event.get("operation", "dispatch"),
                "group": group["name"],
                "status": result.get("status", "completed"),
            }
        )
    )
    return result
