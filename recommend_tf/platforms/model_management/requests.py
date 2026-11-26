"""Freeze requests before launch. One daily slot survives configuration changes."""

import copy
from datetime import date, timedelta

from .contracts import REQUEST, binding_for, exact, identifier, under
from .storage import Journal, conflict


def ready_for(s3, group, model, cutoff):
    import json

    from recommend_tf.train.comm.datasvr.daily_window import window

    from ..sagemaker.publish_model import read

    days = window(model["config"], cutoff)
    source = group["data_source"]
    if source["kind"] == "ready_manifest":
        from ..sagemaker.ready_event import build_ready

        return build_ready(
            s3,
            {
                **source,
                "training_days": len(days["train"]),
                "validation_days": len(days["validation"]),
            },
            date.fromisoformat(cutoff),
        )
    refs = []
    for day in days["train"] + days["validation"]:
        pointer = json.loads(read(s3, source["uri"].rstrip("/") + "/date=" + day + "/_SUCCESS")[0])
        refs.append({**pointer, "date": day})
    return {"version": "idea-mapped-ready/v1", "cutoff": cutoff, "sources": refs}


def next_cutoff(state, model):
    if state["active"]:
        return None
    if state["history"] and state["history"][-1]["stage"] not in {"deployed", "rolled_back"}:
        return None  # A terminal failure needs an explicit retry, never a silent new run.
    last = state["high_water"]
    return (
        (date.fromisoformat(last) + timedelta(days=1)).isoformat()
        if last
        else model["first_cutoff"]
    )


def track_pending(documents, name, slot_uri):
    """Index before admitting the slot so a lost response cannot hide runnable work."""
    journal = Journal(documents, documents.uri("pending/" + name + ".json"))
    journal.value = journal.value or {"slots": []}
    if slot_uri not in journal.value["slots"]:
        journal.value["slots"].append(slot_uri)
        journal.save()


def enqueue(
    documents,
    group,
    registry,
    name,
    *,
    cutoff,
    bootstrap_run=None,
    rollback_version=None,
    operation_id=None,
    retry_id="",
    ready_builder=ready_for,
):
    model = registry["models"][name]
    if "adoption" in model:
        owner, _ = documents.read(documents.uri("ownership/" + name + ".json"))
        if not owner or owner["status"] != "managed":
            raise ValueError("legacy admission handover incomplete")
    if model["admission"] != "managed":
        raise ValueError("legacy model admission has not been handed over")
    date.fromisoformat(cutoff)
    if bootstrap_run and rollback_version:
        raise ValueError("bootstrap and rollback are exclusive")
    kind = "rollback" if rollback_version else "bootstrap" if bootstrap_run else "daily"
    if kind != "daily" and not operation_id:
        raise ValueError("explicit operation_id required")
    if operation_id:
        identifier(operation_id)
    slot = name + "/" + (cutoff if kind == "daily" else kind + "-" + operation_id)
    slot_uri = documents.uri("slots/" + slot + ".json")
    prior, _ = documents.read(slot_uri)
    if prior:
        if retry_id and prior.get("retry_id") != retry_id:
            raise ValueError("cancel/rebuild the existing slot explicitly before retry")
        return prior
    binding = binding_for(group, name, model)
    sealed = None
    if bootstrap_run:
        under(bootstrap_run, group.get("read_prefixes", []))
        import json

        from ..sagemaker.publish_model import read

        sealed = json.loads(read(documents.s3, bootstrap_run.rstrip("/") + "/COMMITTED.json")[0])
        if sealed.get("version") != "doku-tf-run/v1":
            raise ValueError("successful sealed training run required")
    ready = (
        {"cutoff": cutoff}
        if bootstrap_run or rollback_version
        else ready_builder(documents.s3, group, model, cutoff)
    )
    registry_pin = documents.immutable("registry-snapshots", registry)
    binding["managed_model"].update(
        registry_uri=registry_pin["uri"], registry_sha256=registry_pin["sha256"]
    )
    pins = {
        k: documents.immutable("frozen/" + k, v)
        for k, v in (("config", model["config"]), ("binding", binding), ("ready", ready))
    }
    value = {
        "schema_version": REQUEST,
        "environment": group["environment"],
        "group": group["name"],
        "model_name": name,
        "config_revision": model["config_revision"],
        "registry": registry_pin,
        "runtime": documents.immutable("groups", group),
        "bootstrap_evidence": documents.immutable("run-inventories", sealed) if sealed else None,
        "pins": pins,
        "cutoff": cutoff,
        "bootstrap_run": bootstrap_run,
        "rollback_version": rollback_version,
        "retry_id": retry_id,
        "slot_uri": slot_uri,
    }
    pin = documents.immutable("requests", value)
    record = {"request": pin, "status": "queued", "retry_id": retry_id}
    track_pending(documents, name, slot_uri)
    try:
        documents.put(slot_uri, record)
    except Exception as error:
        if not conflict(error):
            raise
        record = documents.read(slot_uri)[0]
    return record


def cancel(documents, slot_uri, expected_request):
    record, etag = documents.read(slot_uri)
    if record["request"] != expected_request or record["status"] not in {"queued", "cancelled"}:
        raise ValueError("only an unstarted matching request can be cancelled")
    if record["status"] != "cancelled":
        documents.put(slot_uri, {**record, "status": "cancelled"}, etag)
    return {"status": "cancelled", "request": expected_request}


def rebuild(documents, group, registry, slot_uri, expected_request, *, retry_id):
    identifier(retry_id)
    old, etag = documents.read(slot_uri)
    if old["request"] != expected_request or old["status"] not in {"cancelled", "failed"}:
        raise ValueError("rebuild requires a cancelled/unstarted or failed request")
    previous = documents.pinned(old["request"])
    from ..sagemaker.publish_model import ReleaseState

    old_binding = documents.pinned(previous["pins"]["binding"])
    if ReleaseState(documents.s3, old_binding["state_uri"], group["kms_key"]).value["active"]:
        raise ValueError("active release must be recovered before rebuilding")
    # Freeze a separate explicit retry slot, then atomically redirect the original daily slot.
    # A dispatcher can only launch a record whose request still points back to its slot.
    fresh = copy.deepcopy(previous)
    name = previous["model_name"]
    model = registry["models"][name]
    binding = binding_for(group, name, model)
    registry_pin = documents.immutable("registry-snapshots", registry)
    binding["managed_model"].update(
        registry_uri=registry_pin["uri"], registry_sha256=registry_pin["sha256"]
    )
    ready = documents.pinned(previous["pins"]["ready"])
    # Re-resolve the window when rebuilding with a different recipe, before making it runnable.
    if not previous["bootstrap_run"] and not previous["rollback_version"]:
        ready = ready_for(documents.s3, group, model, previous["cutoff"])
    fresh.update(
        config_revision=model["config_revision"],
        registry=registry_pin,
        runtime=documents.immutable("groups", group),
        retry_id=retry_id,
        pins={
            k: documents.immutable("frozen/" + k, v)
            for k, v in (("config", model["config"]), ("binding", binding), ("ready", ready))
        },
    )
    pin = documents.immutable("requests", fresh)
    record = {
        "request": pin,
        "status": "queued",
        "retry_id": retry_id,
        "previous_request": old["request"],
    }
    track_pending(documents, name, slot_uri)
    documents.put(slot_uri, record, etag)
    return record


def validate_request(documents, group, pin):
    under(pin["uri"], [documents.uri("requests")])
    request = documents.pinned(pin)
    exact(
        request,
        (
            "schema_version",
            "environment",
            "group",
            "model_name",
            "config_revision",
            "registry",
            "runtime",
            "bootstrap_evidence",
            "pins",
            "cutoff",
            "bootstrap_run",
            "rollback_version",
            "retry_id",
            "slot_uri",
        ),
    )
    if (
        request["schema_version"] != REQUEST
        or request["environment"] != group["environment"]
        or request["group"] != group["name"]
    ):
        raise ValueError("request ownership mismatch")
    from .contracts import validate_group, validate_registry

    under(request["runtime"]["uri"], [documents.uri("groups")])
    frozen_group = validate_group(documents.pinned(request["runtime"]))
    if any(
        frozen_group[k] != group[k] for k in ("name", "environment", "region", "storage_prefix")
    ):
        raise ValueError("frozen runtime crosses group boundary")
    group = frozen_group

    under(request["registry"]["uri"], [documents.uri("registry-snapshots")])
    registry = validate_registry(documents.pinned(request["registry"]), group)
    model = registry["models"][request["model_name"]]
    if model["admission"] != "managed" or request["config_revision"] != model["config_revision"]:
        raise ValueError("request registration differs")
    values = {"group": group}
    for name in ("config", "binding", "ready"):
        under(request["pins"][name]["uri"], [documents.uri("frozen/" + name)])
        values[name] = documents.pinned(request["pins"][name])
    expected = binding_for(group, request["model_name"], model)
    expected["managed_model"].update(
        registry_uri=request["registry"]["uri"], registry_sha256=request["registry"]["sha256"]
    )
    if (
        values["binding"] != expected
        or values["config"] != model["config"]
        or values["ready"]["cutoff"] != request["cutoff"]
    ):
        raise ValueError("request does not match its frozen registration")
    under(request["slot_uri"], [documents.uri("slots/" + request["model_name"])])
    if request["bootstrap_run"]:
        under(request["bootstrap_run"], group.get("read_prefixes", []))
        under(request["bootstrap_evidence"]["uri"], [documents.uri("run-inventories")])
        values["bootstrap_evidence"] = documents.pinned(request["bootstrap_evidence"])
        if values["bootstrap_evidence"].get("version") != "doku-tf-run/v1":
            raise ValueError("sealed bootstrap inventory required")
    elif request["bootstrap_evidence"] is not None:
        raise ValueError("bootstrap inventory without source run")
    return request, values
