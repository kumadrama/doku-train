"""Recoverable S3-authoritative Redis distribution and coordinated retirement.

A persistent operation serializes pointer changes and deletion. It is deliberately
not a lease: resumption requires evidence that the previous executor has stopped.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from recommend.train.comm.artifact_store import ConflictError
from recommend.train.comm.publication import Registry, encode, instant, verify_objects
from recommend.train.comm.redis_artifact_store import RedisTransport, identifier


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class Distribution:
    def __init__(
        self,
        registry: Registry,
        redis: RedisTransport,
        *,
        environment: str,
        max_package_bytes: int,
        max_model_bytes: int,
    ) -> None:
        if not 0 < max_package_bytes <= max_model_bytes:
            raise ValueError("positive package/model budgets required")
        self.registry = registry
        self.redis = redis
        self.environment = identifier(environment)
        self.prefix = f"doku:model:{{{self.environment}:{identifier(registry.name)}}}:"
        self.catalog_key = self.prefix + "catalog"
        self.max_package_bytes = max_package_bytes
        self.max_model_bytes = max_model_bytes
        self.owner = uuid.uuid4().hex

    def _state(self) -> dict[str, Any]:
        state = self.registry.read()[0]
        d = state.get("distribution")
        if d and d["prefix"] != self.prefix:
            raise ValueError("distribution environment mismatch")
        return state

    def enable(self) -> None:
        """v2 fences v1 binaries; never migrate an active producer/deleter."""
        if self._state()["format"] == "doku-publication/v2":
            return
        if self.redis.get(self.catalog_key) is not None:
            raise ConflictError("unowned Redis catalog; explicit reconciliation required")

        def migrate(state: dict[str, Any]) -> None:
            if state["format"] == "doku-publication/v2":
                if state["distribution"]["prefix"] != self.prefix:
                    raise ConflictError("distribution binding changed")
                return
            if state.get("active_run") or any(
                v["state"] == "deleting" for v in state["versions"].values()
            ):
                raise ConflictError("cannot migrate active producer/deleter")
            state["format"] = "doku-publication/v2"
            state["distribution"] = {
                "prefix": self.prefix,
                "epoch": uuid.uuid4().hex,
                "generation": 0,
                "catalog": None,
                "operation": None,
                "versions": {},
                "protocol_ready": False,
            }

        self.registry.update(migrate)

    def protocol_ready(self) -> None:
        def mark(state: dict[str, Any]) -> None:
            if state["distribution"]["operation"]:
                raise ConflictError("operation pending")
            state["distribution"]["protocol_ready"] = True

        self._consistent()
        self.registry.update(mark)

    def _consistent(self) -> dict[str, Any]:
        state = self._state()
        if state["format"] != "doku-publication/v2":
            raise ValueError("Redis distribution has not been initialized")
        d = state["distribution"]
        if d["operation"]:
            raise ConflictError("distribution operation pending; resume explicitly")
        expected = encode(d["catalog"]) if d["catalog"] is not None else None
        if self.redis.get(self.catalog_key) != expected:
            raise ConflictError("Redis catalog lost or changed; rebuild epoch explicitly")
        return state

    def _bundle(
        self, version: str, state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        identifier(version)
        entry = state["versions"][version]
        if not entry["published"] or entry["state"] not in {"available", "draining"}:
            raise ValueError("version is not a retained published package")
        base = f"{self.registry.name}/{entry['date']}/{version}/"
        raw = self.registry.store.get(base + "manifest.json")[0]
        if sha(raw) != entry["manifest_sha256"]:
            raise ValueError("source manifest checksum mismatch")
        manifest = json.loads(raw)
        if (
            manifest["format"] != "doku-tree-artifact/v1"
            or manifest["model_name"] != self.registry.name
            or manifest["model_version"] != version
            or manifest["date"] != entry["date"]
            or manifest["model_type"] != "rerank_tree"
            or manifest["parity"]["status"] != "passed"
        ):
            raise ValueError("source manifest identity mismatch")
        required = {"feature-contract.json", "fitted-feature-state/state.json"}
        if not manifest["tasks"]:
            raise ValueError("missing task models")
        for task in manifest["tasks"].values():
            path = task["file"]
            if not path.startswith("models/") or not path.endswith(".ubj"):
                raise ValueError("invalid task model path")
            required.add(path)
        prefix = self.prefix + f"v:{version}:"
        payloads = {prefix + "file:manifest.json": raw}
        files: dict[str, Any] = {}
        total = len(raw)
        for path in sorted(required):
            self.registry.store.key(path)
            expected = manifest["files"][path]
            if entry["objects"].get(base + path) != expected:
                raise ValueError("source inventory mismatch")
            body = self.registry.store.get(base + path)[0]
            total += len(body)
            if total > self.max_package_bytes:
                raise ValueError("serving package exceeds budget")
            if sha(body) != expected:
                raise ValueError("serving file checksum mismatch")
            key = prefix + "file:" + path
            payloads[key] = body
            files[path] = {"key": key, "bytes": len(body), "sha256": expected}
        files["manifest.json"] = {
            "key": prefix + "file:manifest.json",
            "bytes": len(raw),
            "sha256": sha(raw),
        }
        descriptor = {
            "format": "doku-redis-artifact/v1",
            "model_name": self.registry.name,
            "model_type": manifest["model_type"],
            "model_version": version,
            "date": entry["date"],
            "manifest_uri": entry["manifest_uri"],
            "manifest_sha256": sha(raw),
            "feature_contract_sha256": manifest["files"]["feature-contract.json"],
            "required_files": files,
        }
        descriptor_raw = encode(descriptor)
        payloads[prefix + "descriptor"] = descriptor_raw
        payloads[prefix + "ready"] = encode({"descriptor_sha256": sha(descriptor_raw)})
        total = sum(map(len, payloads.values()))
        if total > self.max_package_bytes:
            raise ValueError("serving package exceeds budget")
        record = {
            "reference": {
                "version": version,
                "manifest_sha256": sha(raw),
                "descriptor_sha256": sha(descriptor_raw),
                "feature_contract_sha256": descriptor["feature_contract_sha256"],
            },
            "objects": {key: sha(body) for key, body in payloads.items()},
            "bytes": total,
            "descriptor_key": (
                f"{self.registry.name}/distribution/{version}/{sha(descriptor_raw)}.json"
            ),
            "descriptor": descriptor,
        }
        return record, payloads

    def _begin(
        self,
        state: dict[str, Any],
        *,
        kind: str,
        versions: list[str],
        target: dict[str, Any],
        expected: bytes | None,
    ) -> dict[str, Any]:
        generation = state["distribution"]["generation"] + 1
        operation_id = uuid.uuid4().hex
        target.update(generation=generation, operation_id=operation_id)
        operation = {
            "id": operation_id,
            "owner": self.owner,
            "kind": kind,
            "versions": versions,
            "target": target,
            "expected_base64": base64.b64encode(expected).decode()
            if expected is not None
            else None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        old_revision = state.get("revision")

        def acquire(latest: dict[str, Any]) -> None:
            if latest.get("revision") != old_revision:
                raise ConflictError("registry changed while planning; retry")
            if latest.get("active_run") or latest["distribution"]["operation"]:
                raise ConflictError("producer/distribution operation active")
            latest["distribution"]["generation"] = generation
            latest["distribution"]["operation"] = operation
            if kind == "drain":
                for version in versions:
                    latest["versions"][version]["state"] = "draining"
                    latest["dates"] = {k: v for k, v in latest["dates"].items() if v != version}
            if kind == "delete":
                for version in versions:
                    latest["versions"][version]["state"] = "deleting"

        self.registry.update(acquire)
        return operation

    @staticmethod
    def _expected(op: dict[str, Any]) -> bytes | None:
        value = op["expected_base64"]
        return base64.b64decode(value, validate=True) if value is not None else None

    def _owned(self, state: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
        current = state["distribution"]["operation"]
        if not current or current["id"] != operation["id"] or current["owner"] != self.owner:
            raise ConflictError("distribution ownership changed")
        return dict(current)

    def sync(
        self, version: str, *, rollback: bool = False, stage_only: bool = False
    ) -> dict[str, Any]:
        self.enable()
        state = self._consistent()
        if rollback and stage_only:
            raise ValueError("rollback cannot be stage-only")
        if not rollback and not stage_only and state["current"] != version:
            raise ValueError("sync must target exact S3 current; use explicit rollback")
        if rollback:
            verify_objects(self.registry.store, state["versions"][version])
        record, _ = self._bundle(version, state)
        d = state["distribution"]
        catalog = (
            copy.deepcopy(d["catalog"])
            if d["catalog"]
            else {
                "format": "doku-redis-catalog/v1",
                "epoch": d["epoch"],
                "current": None,
                "dates": {},
            }
        )
        # Idempotent calls verify all bytes, not merely the ready marker.
        if not rollback and not stage_only and catalog.get("current") == record["reference"]:
            self.verify(version, state)
            return {"status": "committed", "model_version": version, "unchanged": True}
        catalog["current"] = record["reference"]
        catalog["dates"][state["versions"][version]["date"]] = record["reference"]
        self._begin(
            state,
            kind="stage" if stage_only else "rollback" if rollback else "sync",
            versions=[version],
            target=catalog,
            expected=encode(d["catalog"]) if d["catalog"] else None,
        )
        return self._execute()

    def resume(
        self, operation_id: str, *, confirmed_stopped: bool, rebuild_redis: bool = False
    ) -> dict[str, Any]:
        if not confirmed_stopped:
            raise ValueError("confirm old executor and outstanding requests stopped before resume")
        snapshot = self._state()
        pending = snapshot["distribution"]["operation"]
        if not pending or pending["id"] != operation_id:
            raise ConflictError("operation does not match")
        actual = self.redis.get(self.catalog_key) if rebuild_redis else None

        def takeover(state: dict[str, Any]) -> None:
            op = state["distribution"]["operation"]
            if not op or op["id"] != operation_id:
                raise ConflictError("operation does not match")
            if rebuild_redis:
                if state.get("revision") != snapshot.get("revision"):
                    raise ConflictError("registry changed while planning recovery")
                d = state["distribution"]
                if op["kind"] == "stage":
                    # Shadow uploads must not become online because Redis was rebuilt.
                    op["target"] = copy.deepcopy(d["catalog"]) or {
                        "format": "doku-redis-catalog/v1",
                        "current": None,
                        "dates": {},
                    }
                d["generation"] += 1
                op["target"].update(
                    epoch=uuid.uuid4().hex, generation=d["generation"], operation_id=op["id"]
                )
                op["expected_base64"] = (
                    base64.b64encode(actual).decode() if actual is not None else None
                )
                op["rebuild_redis"] = True
            op["owner"] = self.owner

        self.registry.update(takeover)
        return self._execute()

    def _finish_without_commit(self, op: dict[str, Any], status: str) -> None:
        def finish(state: dict[str, Any]) -> None:
            self._owned(state, op)
            state["distribution"]["last_operation"] = {
                "id": op["id"],
                "kind": op["kind"],
                "status": status,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            state["distribution"]["operation"] = None

        self.registry.update(finish)

    def cancel(self, operation_id: str, *, confirmed_stopped: bool) -> dict[str, Any]:
        """Cancel only an uncommitted upload; lifecycle mutations must be resumed."""
        if not confirmed_stopped:
            raise ValueError("confirm old executor and outstanding requests stopped before cancel")
        state = self._state()
        op = state["distribution"]["operation"]
        if (
            not op
            or op["id"] != operation_id
            or op["kind"] not in {"sync", "stage"}
            or op.get("rebuild_redis")
        ):
            raise ConflictError("only pending sync/stage uploads can be cancelled; resume instead")
        expected = self._expected(op)
        if self.redis.get(self.catalog_key) != expected:
            raise ConflictError("catalog already committed or unknown; resume instead")

        def takeover(latest: dict[str, Any]) -> None:
            if latest.get("revision") != state.get("revision"):
                raise ConflictError("operation changed while cancelling")
            latest["distribution"]["operation"]["owner"] = self.owner

        self.registry.update(takeover)
        self._finish_without_commit(op, "cancelled")
        return {"status": "cancelled", "operation_id": operation_id}

    def rebuild(self, *, confirmed_stopped: bool) -> dict[str, Any]:
        if not confirmed_stopped:
            raise ValueError("confirm old writers and outstanding requests stopped before rebuild")
        state = self._state()
        d = state["distribution"]
        if d["operation"]:
            raise ConflictError("recover pending operation before rebuilding")
        # Rebuild only retained authoritative references, never Redis backup contents.
        if state["current"] is None:
            raise ValueError("no published version to rebuild")
        versions = sorted(set(state["dates"].values()) | {state["current"]})
        refs = {v: self._bundle(v, state)[0]["reference"] for v in versions}
        target = {
            "format": "doku-redis-catalog/v1",
            "epoch": uuid.uuid4().hex,
            "current": refs[state["current"]],
            "dates": {day: refs[v] for day, v in state["dates"].items()},
        }
        self._begin(
            state,
            kind="rebuild",
            versions=versions,
            target=target,
            expected=self.redis.get(self.catalog_key),
        )
        return self._execute()

    def verify(self, version: str, state: dict[str, Any] | None = None) -> None:
        state = state or self._state()
        record = state["distribution"]["versions"][version]
        for key, expected in record["objects"].items():
            self._key_owned(version, key)
            body = self.redis.get(key)
            if body is None or sha(body) != expected:
                raise ValueError("Redis package missing or corrupt; rebuild required")

    def _key_owned(self, version: str, key: str) -> None:
        if not key.startswith(self.prefix + "v:" + identifier(version) + ":"):
            raise ValueError("unowned Redis key")

    def _upload(self, version: str, op: dict[str, Any]) -> None:
        state = self._state()
        self._owned(state, op)
        record, payloads = self._bundle(version, state)

        def register(latest: dict[str, Any]) -> None:
            self._owned(latest, op)
            records = latest["distribution"]["versions"]
            old = records.get(version)
            if old and old["objects"] != record["objects"]:
                raise ConflictError("distribution version collision")
            total = sum(
                r["bytes"] for v, r in records.items() if v != version and not r.get("deleted")
            )
            if total + record["bytes"] > self.max_model_bytes:
                raise ValueError("Redis model retention exceeds budget")
            if not old:
                records[version] = record

        self.registry.update(register)
        self.registry.store.immutable(record["descriptor_key"], encode(record["descriptor"]))
        for key in sorted(payloads, key=lambda k: (k.endswith(":ready"), k)):
            self._owned(self._state(), op)
            if op["kind"] == "rebuild" or op.get("rebuild_redis"):
                existing = self.redis.get(key)
                if existing is not None and existing != payloads[key]:
                    # Explicit recovery, with old writers fenced, repairs corrupt cache bytes.
                    self.redis.delete(key)
            self.redis.immutable(key, payloads[key])
        self.verify(version)

    def _execute(self) -> dict[str, Any]:
        state = self._state()
        op = state["distribution"]["operation"]
        self._owned(state, op)
        kind = op["kind"]
        if op.get("rebuild_redis"):
            refs = list(op["target"]["dates"].values())
            if op["target"]["current"]:
                refs.append(op["target"]["current"])
            for version in sorted({ref["version"] for ref in refs}):
                self._upload(version, op)
            if kind == "delete":
                # Retire a stale/restored catalog BEFORE resuming any destructive operations.
                self._write_catalog(op)
        if kind == "delete":
            self._delete(op)
        else:
            if kind in {"sync", "rollback", "rebuild", "stage"}:
                for version in op["versions"]:
                    self._upload(version, op)
            if kind == "stage":
                if op.get("rebuild_redis"):
                    self._commit(op)
                else:
                    self._finish_without_commit(op, "staged")
                return {"status": "staged", "operation_id": op["id"]}
            if kind == "rollback":
                self.registry.publish(
                    op["versions"][0], "redis-rollback", rollback=True, control=self.owner
                )
            self._commit(op)
        return {"status": "committed", "operation_id": op["id"], "kind": kind}

    def _write_catalog(self, op: dict[str, Any]) -> None:
        state = self._state()
        self._owned(state, op)
        ready: dict[str, bytes] = {}
        refs = list(op["target"]["dates"].values())
        if op["target"]["current"]:
            refs.append(op["target"]["current"])
        for ref in refs:
            version = ref["version"]
            if state["versions"][version]["state"] != "available":
                raise ConflictError("catalog references unavailable version")
            self.verify(version, state)
            ready[self.prefix + f"v:{version}:ready"] = encode(
                {"descriptor_sha256": ref["descriptor_sha256"]}
            )
        self.redis.commit(
            self.catalog_key,
            self._expected(op),
            encode(op["target"]),
            ready,
        )
        if self.redis.get(self.catalog_key) != encode(op["target"]):
            raise ConflictError("Redis commit outcome unknown")

    def _commit(self, op: dict[str, Any]) -> None:
        self._write_catalog(op)
        now = datetime.now(UTC).isoformat()

        def complete(latest: dict[str, Any]) -> None:
            self._owned(latest, op)
            d = latest["distribution"]
            old = d["catalog"]
            if old:
                pairs = [(old["current"], op["target"]["current"])]
                pairs.extend(
                    (ref, op["target"]["dates"].get(day)) for day, ref in old["dates"].items()
                )
                for before, after in pairs:
                    if before and before != after:
                        d["versions"][before["version"]]["replaced_at"] = now
            d["catalog"] = op["target"]
            d["epoch"] = op["target"]["epoch"]
            if op["kind"] == "drain":
                for version in op["versions"]:
                    latest["versions"][version]["drain_started_at"] = now
            d["last_operation"] = {"id": op["id"], "kind": op["kind"], "completed_at": now}
            d["operation"] = None

        self.registry.update(complete)

    def cleanup(self, *, apply: bool = False, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("timezone required")
        initial = self._state()
        if initial["format"] != "doku-publication/v2":
            return {
                "dry_run": True,
                "reason": "distribution_not_initialized",
                "plan": [
                    {"version": v, "action": "distribution_not_initialized"}
                    for v in initial["versions"]
                ],
            }
        state = (
            self._consistent()
        )  # Unknown/missing Redis blocks deletion, even with S3 pins empty.
        d = state["distribution"]
        catalog = d["catalog"]
        current = catalog["current"]["version"] if catalog and catalog["current"] else None
        plan = []
        for version, entry in state["versions"].items():
            record = d["versions"].get(version, {})
            replaced = [
                instant(t) for t in [entry.get("replaced_at"), record.get("replaced_at")] if t
            ]
            if version in {state["current"], current}:
                action = "current"
            elif state.get("active_run") == entry["producer"]:
                action = "producer_active"
            elif entry["state"] == "deleted":
                action = "deleted"
            elif now - instant(entry["created_at"]) <= timedelta(days=30):
                action = "younger_than_30_days"
            elif replaced and now - max(replaced) < timedelta(hours=24):
                action = "replacement_grace"
            elif entry["state"] in {"available", "uploading"}:
                action = "drain"
            elif entry["pins"]:
                action = "active_pins"
            elif now - instant(entry["drain_started_at"]) < timedelta(hours=24):
                action = "drain_grace"
            else:
                action = "delete"
            plan.append({"version": version, "action": action})
        enabled = apply and state["consumer_protocol_ready"] and d["protocol_ready"]
        result = {
            "dry_run": not enabled,
            "plan": plan,
            "consumer_protocol_ready": state["consumer_protocol_ready"],
            "distribution_protocol_ready": d["protocol_ready"],
        }
        if not enabled:
            return result
        # One durable batch per call. Do not hold ownership throughout the 24-hour drain.
        for kind in ["delete", "drain"]:
            versions = [s["version"] for s in plan if s["action"] == kind]
            if not versions:
                continue
            target = (
                copy.deepcopy(catalog)
                if catalog
                else {
                    "format": "doku-redis-catalog/v1",
                    "epoch": d["epoch"],
                    "current": None,
                    "dates": {},
                }
            )
            target["dates"] = {
                day: ref for day, ref in target["dates"].items() if ref["version"] not in versions
            }
            self._begin(
                state,
                kind=kind,
                versions=versions,
                target=target,
                expected=encode(catalog) if catalog else None,
            )
            self._execute()
            result["executed"] = kind
            break
        return result

    def _delete_catalog_guard(self, op: dict[str, Any], version: str) -> None:
        actual = self.redis.get(self.catalog_key)
        expected = self._expected(op)
        if actual not in [expected, encode(op["target"])]:
            raise ConflictError("Redis catalog changed during deletion")
        if actual:
            catalog = json.loads(actual)
            refs = [catalog["current"], *catalog["dates"].values()]
            if any(ref and ref["version"] == version for ref in refs):
                raise ConflictError("Redis catalog still references deletion target")

    def _delete(self, op: dict[str, Any]) -> None:
        for version in op["versions"]:
            state = self._state()
            self._owned(state, op)
            entry = state["versions"][version]
            self._delete_catalog_guard(op, version)
            if (
                entry["state"] not in {"deleting", "deleted"}
                or entry["pins"]
                or state["current"] == version
                or version in state["dates"].values()
            ):
                raise ConflictError("version deletion protection changed")
            record = state["distribution"]["versions"].get(version)
            if record:
                for key in sorted(record["objects"], key=lambda k: (not k.endswith(":ready"), k)):
                    self._delete_catalog_guard(op, version)
                    self._key_owned(version, key)
                    self.redis.delete(key)
                if any(self.redis.get(k) is not None for k in record["objects"]):
                    raise ConflictError("Redis deletion not complete")
                descriptor_key = record["descriptor_key"]
                if not descriptor_key.startswith(f"{self.registry.name}/distribution/{version}/"):
                    raise ValueError("unowned descriptor key")
                self._delete_catalog_guard(op, version)
                self.registry.store.delete(descriptor_key)
            for key in entry["objects"]:
                self._delete_catalog_guard(op, version)
                if not key.startswith(f"{self.registry.name}/{entry['date']}/{version}/"):
                    raise ValueError("unowned S3 object")
                self.registry.store.delete(key)

            def deleted(latest: dict[str, Any], v: str = version) -> None:
                self._owned(latest, op)
                latest["versions"][v]["state"] = "deleted"
                if v in latest["distribution"]["versions"]:
                    latest["distribution"]["versions"][v]["deleted"] = True

            self.registry.update(deleted)
        # Keep Redis and S3 catalog generations in sync even when only removing bytes.
        self._commit(op)
