"""Plan/apply registry changes using existing AppConfig CAS and a durable single writer."""

import copy
import json

from ..sagemaker.idea_appconfig import Publisher
from .contracts import canonical, digest, exact, validate_model, validate_registry
from .storage import Journal, conflict


class Registry:
    def __init__(self, documents, group, control, data):
        self.docs, self.group = documents, group
        self.publisher = Publisher(control, data, group["registry"], label_prefix="model-registry")
        self.profile = group["registry"]["profile_id"]

    def read(self):
        return validate_registry(json.loads(self.publisher.read(self.profile)), self.group)

    def plan(self, action, name, payload, expected):
        before = self.read()
        if digest(before) != expected:
            raise ValueError("registry changed; refresh before planning")
        after = copy.deepcopy(before)
        models = after["models"]
        if action == "register":
            if name in models:
                raise ValueError("model already registered")
            exact(payload, ("config", "service", "first_cutoff"))
            models[name] = {
                **payload,
                "group": self.group["name"],
                "config_revision": digest(payload["config"]),
                "daily_enabled": False,
                "admission": "managed",
            }
        elif action in {"configure", "restore_config"}:
            exact(payload, ("config",))
            model = models[name]
            model["config"] = payload["config"]
            model["config_revision"] = digest(payload["config"])
        elif action == "daily":
            exact(payload, ("enabled",))
            if type(payload["enabled"]) is not bool:
                raise ValueError("daily enabled must be boolean")
            models[name]["daily_enabled"] = payload["enabled"]
        else:
            raise ValueError("unsupported registry operation")
        validate_model(name, models[name], self.group)
        after["revision"] += 1
        validate_registry(after, self.group)
        intent = {
            "schema_version": "doku-model-change/v1",
            "action": action,
            "model_name": name,
            "before": before,
            "after": after,
        }
        pin = self.docs.immutable("plans", intent)
        return {
            "plan": pin,
            "before_revision": before["revision"],
            "after_revision": after["revision"],
            "changes": changes(before["models"].get(name), models[name]),
            "affects": "future_requests_only",
        }

    def apply(self, pin):
        intent = self.docs.pinned(pin)
        before, after = intent["before"], intent["after"]
        validate_registry(before, self.group)
        validate_registry(after, self.group)
        operation = pin["sha256"]
        receipt_uri = self.docs.uri("changes/" + operation + ".json")
        prior, _ = self.docs.read(receipt_uri)
        if prior:
            journal = Journal(self.docs, self.docs.uri("registry-publication.json"))
            if journal.value and (journal.value.get("active") or {}).get("operation") == operation:
                journal.value["active"] = None
                journal.save()
            return prior
        journal = Journal(self.docs, self.docs.uri("registry-publication.json"))
        if journal.value is None:
            journal.value = {"active": None}
        if journal.value["active"] is None:
            if self.read() != before:
                raise ValueError("stale registry plan")
            raw = self.publisher.read(self.profile)
            if json.loads(raw) != before:
                raise ValueError("registry changed during admission")
            journal.value["active"] = {
                "operation": operation,
                "online": {},
                "before_raw": raw.decode(),
            }
            journal.save()
        if journal.value["active"]["operation"] != operation:
            raise ValueError("another registry publication is active; resume it first")
        self.docs.immutable("registry-snapshots", before)
        self.docs.immutable("registry-snapshots", after)
        if not self.publisher.step(
            journal,
            "registry",
            self.profile,
            journal.value["active"]["before_raw"].encode(),
            canonical(after),
        ):
            return {"status": "pending", "operation": operation, "plan": pin}
        result = {
            "status": "complete",
            "operation": operation,
            "registry_sha256": digest(after),
            "revision": after["revision"],
        }
        try:
            self.docs.put(receipt_uri, result)
        except Exception as error:
            if not conflict(error) or self.docs.read(receipt_uri)[0] != result:
                raise
        journal.value["active"] = None
        journal.save()
        return result


def changes(before, after, path=""):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            item
            for k in sorted(before.keys() | after.keys())
            for item in changes(before.get(k), after.get(k), path + "/" + k)
        ]
    return [{"path": path, "before": before, "after": after}]
