"""Explicit legacy admission handover; never updates an Endpoint or consumer profile."""

import copy
import hashlib
import json
import tarfile
import time
from datetime import date, timedelta

from ..sagemaker.publish_model import location, read
from .contracts import binding_for, digest, under, validate_model
from .registry import changes
from .storage import Journal


def target_snapshot(events, ecs, rule):
    targets = events.list_targets_by_rule(Rule=rule)["Targets"]
    if len(targets) != 1 or "EcsParameters" not in targets[0]:
        raise ValueError("one dedicated legacy worker target required")
    td = ecs.describe_task_definition(
        taskDefinition=targets[0]["EcsParameters"]["TaskDefinitionArn"]
    )["taskDefinition"]
    containers = [c for c in td["containerDefinitions"] if c["name"] == "pipeline"]
    if len(containers) != 1:
        raise ValueError("unambiguous legacy pipeline container required")
    env = {v["name"]: v["value"] for v in containers[0].get("environment", [])}
    retry = targets[0].get("RetryPolicy", {})
    if type(retry.get("MaximumEventAgeInSeconds")) is not int:
        raise ValueError("legacy delivery retry window must be explicit before adoption")
    return {
        "targets": targets,
        "task_definition": td["taskDefinitionArn"],
        "config": {"uri": env["DOKU_CONFIG_URI"], "sha256": env["DOKU_CONFIG_SHA256"]},
        "binding": {"uri": env["DOKU_BINDING_URI"], "sha256": env["DOKU_BINDING_SHA256"]},
    }


def legacy_pin(documents, group, pin):
    under(pin["uri"], group.get("read_prefixes", []) + group.get("adoption_prefixes", []))
    raw = read(documents.s3, pin["uri"])[0]
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("legacy input digest differs")
    return json.loads(raw)


def serving_contract(s3, uri, limit):
    """Stream metadata from the existing sealed artifact without extracting files."""
    prefix = uri.rsplit("/", 1)[0]
    manifest = json.loads(read(s3, prefix + "/manifest.json")[0])
    bucket, key = location(uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    if response["ContentLength"] > limit:
        raise ValueError("adoption artifact exceeds explicit download budget")
    found = {}
    with response["Body"] as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("doku-contract/")
            if name not in {"signature.json", "feature-config.json"}:
                continue
            if not member.isfile() or name in found or member.size > 2 * 1024 * 1024:
                raise ValueError("invalid serving contract member")
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                raise ValueError("serving contract differs from sealed manifest")
            found[name] = json.loads(raw)
    if set(found) != {"signature.json", "feature-config.json"}:
        raise ValueError("existing artifact lacks serving contract")
    found["signature.json"].pop("model_version", None)
    return {"signature": found["signature.json"], "feature_config": found["feature-config.json"]}


def plan(documents, group, registry, name, rule, *, events, ecs, sm):
    if rule not in group.get("legacy_rules", []):
        raise ValueError("legacy rule not enrolled by platform")
    if name in registry["models"]:
        owner, _ = documents.read(documents.uri("ownership/" + name + ".json"))
        if not owner or owner["status"] != "legacy" or "adoption" not in registry["models"][name]:
            raise ValueError("model already registered")
    snapshot = target_snapshot(events, ecs, rule)
    config = legacy_pin(documents, group, snapshot["config"])
    binding = legacy_pin(documents, group, snapshot["binding"])
    if (
        config["model_name"] != name
        or binding.get("model_name", binding.get("monitoring", {}).get("model_name")) != name
        or binding["environment"] != group["environment"]
    ):
        raise ValueError("legacy model ownership mismatch")
    state, _ = documents.read(binding["state_uri"])
    if not state or not state.get("online"):
        raise ValueError("adoption requires a published legacy model")
    endpoint = binding["deployment"]["endpoint_name"]
    observed = sm.describe_endpoint(EndpointName=endpoint)
    if (
        observed["EndpointStatus"] != "InService"
        or observed["EndpointConfigName"] != state["online"]["endpoint_config"]
    ):
        raise ValueError("legacy live Endpoint and journal disagree")
    release = next(
        r
        for r in reversed(state["history"])
        if r.get("model_version") == state["online"]["model_version"] and r.get("model_uri")
    )
    endpoint_config = sm.describe_endpoint_config(EndpointConfigName=observed["EndpointConfigName"])
    variants = endpoint_config["ProductionVariants"]
    if len(variants) != 1:
        raise ValueError("single-variant managed service required")
    deployed = sm.describe_model(ModelName=variants[0]["ModelName"])
    if deployed["PrimaryContainer"]["ModelDataUrl"] != release["model_uri"]:
        raise ValueError("actual serving artifact differs from release journal")
    contract = documents.immutable(
        "adopted-contracts",
        serving_contract(documents.s3, release["model_uri"], binding["download_limit_bytes"]),
    )
    ingress = []
    for ref in group.get("legacy_ingress", {}).get(rule, []):
        matches = [
            target
            for target in events.list_targets_by_rule(Rule=ref["rule"])["Targets"]
            if target["Id"] == ref["target_id"]
        ]
        if len(matches) != 1:
            raise ValueError("legacy ingress target cannot be verified")
        grace = (
            matches[0]
            .get("RetryPolicy", {})
            .get("MaximumEventAgeInSeconds", ref.get("retry_window_seconds"))
        )
        if type(grace) is not int or grace < 0:
            raise ValueError("ingress retry drain window must be explicit")
        ingress.append({"rule": ref["rule"], "target": matches[0], "drain_seconds": grace})
    evidence = documents.immutable(
        "adoption-evidence",
        {
            "rule": rule,
            "ingress": ingress,
            "snapshot": snapshot,
            "state_sha256": digest(state),
            "endpoint": endpoint,
            "endpoint_config": observed["EndpointConfigName"],
            "contract": contract,
            "high_water": state["high_water"],
        },
    )
    service = {
        "outputs": binding["outputs"],
        "probe": binding["probe"],
        "identity_signature": "serving_default"
        if "idea_appconfig" in binding
        else "serving_with_model_version",
    }
    model = {
        "group": group["name"],
        "config": config,
        "config_revision": digest(config),
        "service": service,
        "daily_enabled": events.describe_rule(Name=rule)["State"] == "ENABLED",
        "admission": "managed",
        "first_cutoff": (date.fromisoformat(state["high_water"]) + timedelta(days=1)).isoformat(),
        "adoption": {"binding": binding, "legacy_rule": rule, "evidence_uri": evidence["uri"]},
    }
    validate_model(name, model, group)
    proposed_binding = binding_for(group, name, model)
    # Runtime provenance may advance with the new controller, but serving/data semantics cannot.
    for key in ("deployment", "outputs", "probe", "state_uri", "work_prefix", "model_prefix"):
        if proposed_binding[key] != binding[key]:
            raise ValueError("adoption would change existing behavior: " + key)
    after = copy.deepcopy(registry)
    after["models"][name] = model
    after["revision"] += 1
    intent = {
        "schema_version": "doku-model-change/v1",
        "action": "adopt",
        "model_name": name,
        "before": registry,
        "after": after,
        "evidence": evidence,
    }
    pin = documents.immutable("plans", intent)
    return {"plan": pin, "changes": changes(None, model), "endpoint_unchanged": True}


def idle(documents, group, evidence, binding, ecs):
    state, _ = documents.read(binding["state_uri"])
    if state["active"] or digest(state) != evidence["state_sha256"]:
        raise ValueError("legacy pipeline changed or is running; replan when idle")
    args = {
        "cluster": group["worker"]["cluster"],
        "family": evidence["snapshot"]["task_definition"].split("/")[-1].rsplit(":", 1)[0],
    }
    observed = set()
    # StopTask changes desiredStatus before the executor actually exits.
    # Default ListTasks only sees desired RUNNING and would miss that interval.
    for desired in ("RUNNING", "STOPPED"):
        query = dict(args, desiredStatus=desired)
        while True:
            page = ecs.list_tasks(**query)
            observed.update(page["taskArns"])
            if not page.get("nextToken"):
                break
            query["nextToken"] = page["nextToken"]
    arns = sorted(observed)
    for start in range(0, len(arns), 100):
        result = ecs.describe_tasks(
            cluster=group["worker"]["cluster"], tasks=arns[start : start + 100]
        )
        if result.get("failures") or any(t["lastStatus"] != "STOPPED" for t in result["tasks"]):
            raise ValueError("legacy executor has not stopped")
    if binding.get("deferred_uri"):
        for uri in documents.list_external(binding["deferred_uri"].rstrip("/") + "/requests/"):
            request, _ = documents.read(uri)
            if request["cutoff"] > state["high_water"]:
                raise ValueError("unconsumed legacy request must finish before handover")


def apply(documents, group, registry_store, pin, *, events, ecs, sm, now=time.time):
    intent = documents.pinned(pin)
    if intent["action"] != "adopt":
        raise ValueError("adoption plan required")
    name = intent["model_name"]
    model = intent["after"]["models"][name]
    evidence = documents.pinned(intent["evidence"])
    rule = evidence["rule"]
    if rule not in group.get("legacy_rules", []):
        raise ValueError("legacy rule not allowed")
    owner = Journal(documents, documents.uri("ownership/" + name + ".json"))
    if owner.value is None or owner.value["status"] == "legacy":
        owner.value = {"operation": pin["sha256"], "status": "preparing"}
        owner.save()
    if owner.value["operation"] != pin["sha256"]:
        raise ValueError("another handover owns this model")
    if owner.value["status"] == "managed":
        return owner.value
    if target_snapshot(events, ecs, rule) != evidence["snapshot"]:
        raise ValueError("legacy target changed")
    # Only an explicitly planned dedicated worker admission is paused, never an upstream rule.
    idle(documents, group, evidence, model["adoption"]["binding"], ecs)
    if "drain_after" not in owner.value:
        events.disable_rule(Name=rule)
        for ingress in evidence.get("ingress", []):
            response = events.remove_targets(Rule=ingress["rule"], Ids=[ingress["target"]["Id"]])
            if response.get("FailedEntryCount"):
                raise RuntimeError("legacy ingress removal failed")
        # Disabling a rule does not prove all previously accepted retries drained.
        windows = [
            t["RetryPolicy"]["MaximumEventAgeInSeconds"] for t in evidence["snapshot"]["targets"]
        ]
        windows += [v["drain_seconds"] for v in evidence.get("ingress", [])]
        owner.value.update(status="legacy_paused", drain_after=now() + max(windows))
        owner.save()
    if "revision_fence" in owner.value:
        from .adoption_fence import verify

        verified = verify(documents, group, evidence, name, ecs)
        if verified != owner.value["revision_fence"]["inventory"]:
            raise ValueError("legacy revision inventory changed after fencing")
        if now() < owner.value["revision_fence"]["ready_after"]:
            return {**owner.value, "waiting": "legacy_revision_fence_propagating"}
    elif now() < owner.value["drain_after"]:
        return {**owner.value, "waiting": "legacy_delivery_retries_draining"}
    if events.describe_rule(Name=rule)["State"] != "DISABLED":
        raise ValueError("legacy admission remains enabled")
    idle(documents, group, evidence, model["adoption"]["binding"], ecs)
    endpoint = sm.describe_endpoint(EndpointName=evidence["endpoint"])
    if (
        endpoint["EndpointStatus"] != "InService"
        or endpoint["EndpointConfigName"] != evidence["endpoint_config"]
    ):
        raise ValueError("live Endpoint changed during adoption")
    result = registry_store.apply(pin)
    if result["status"] != "complete":
        return {**owner.value, "registry": result}
    # Registry membership alone cannot admit this model: dispatcher also requires this CAS.
    idle(documents, group, evidence, model["adoption"]["binding"], ecs)
    owner.value["status"] = "managed"
    owner.save()
    return owner.value


def abort(documents, group, pin, *, events):
    """Release the old admission only while the new owner has never been activated."""
    intent = documents.pinned(pin)
    if intent["action"] != "adopt":
        raise ValueError("adoption plan required")
    name = intent["model_name"]
    owner = Journal(documents, documents.uri("ownership/" + name + ".json"))
    if (
        not owner.value
        or owner.value["operation"] != pin["sha256"]
        or owner.value["status"] == "managed"
    ):
        raise ValueError("activated handover cannot be aborted; drain managed requests first")
    if "revision_fence" in owner.value:
        raise ValueError("retired legacy revisions require an explicit replacement release")
    evidence = documents.pinned(intent["evidence"])
    if evidence["rule"] not in group.get("legacy_rules", []):
        raise ValueError("legacy rule not allowed")
    # Keep the durable legacy owner even if an in-flight AppConfig publication later completes.
    owner.value["status"] = "legacy"
    owner.value.pop("drain_after", None)
    owner.save()
    for ingress in evidence.get("ingress", []):
        response = events.put_targets(Rule=ingress["rule"], Targets=[ingress["target"]])
        if response.get("FailedEntryCount"):
            raise RuntimeError("legacy ingress restoration failed")
    if intent["after"]["models"][name]["daily_enabled"]:
        events.enable_rule(Name=evidence["rule"])
    return owner.value


def fence(documents, group, pin, *, events, ecs, now=time.time):
    """Record a server-observed retirement barrier; never edits the original drain."""
    from .adoption_fence import PROPAGATION_SECONDS, verify

    intent = documents.pinned(pin)
    if intent["action"] != "adopt":
        raise ValueError("adoption plan required")
    name = intent["model_name"]
    evidence = documents.pinned(intent["evidence"])
    owner = Journal(documents, documents.uri("ownership/" + name + ".json"))
    if (
        not owner.value
        or owner.value["operation"] != pin["sha256"]
        or owner.value["status"] != "legacy_paused"
    ):
        raise ValueError("matching paused adoption required")
    if evidence["rule"] not in group.get("legacy_rules", []):
        raise ValueError("legacy rule not allowed")
    if events.describe_rule(Name=evidence["rule"])["State"] != "DISABLED":
        raise ValueError("legacy admission remains enabled")
    for ingress in evidence.get("ingress", []):
        targets = events.list_targets_by_rule(Rule=ingress["rule"])["Targets"]
        if any(t["Id"] == ingress["target"]["Id"] for t in targets):
            raise ValueError("legacy ingress remains enabled")
    idle(documents, group, evidence, intent["after"]["models"][name]["adoption"]["binding"], ecs)
    inventory = verify(documents, group, evidence, name, ecs)
    if "revision_fence" not in owner.value:
        observed_at = now()
        owner.value["revision_fence"] = {
            "inventory": inventory,
            "observed_at": observed_at,
            "ready_after": observed_at + PROPAGATION_SECONDS,
        }
        owner.save()
    elif owner.value["revision_fence"]["inventory"] != inventory:
        raise ValueError("legacy revision inventory changed after fencing")
    return owner.value
