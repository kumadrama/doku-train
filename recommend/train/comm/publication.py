"""Single-object CAS registry for publication, persistent pins, rollback and retirement."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from recommend.train.comm.artifact_store import ConflictError, ObjectStore


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must have timezone")
    return parsed


class Registry:
    def __init__(self, store: ObjectStore, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("invalid model_name")
        self.store = store
        self.name = name
        self.key = name + "/publication.json"

    def read(self) -> tuple[dict[str, Any], str | None]:
        try:
            raw, etag = self.store.get(self.key)
            value = json.loads(raw)
            if value["format"] != "doku-publication/v1" or value["model_name"] != self.name:
                raise ValueError("invalid publication registry")
            return value, etag
        except FileNotFoundError:
            return {
                "format": "doku-publication/v1",
                "model_name": self.name,
                "current": None,
                "dates": {},
                "versions": {},
                "runs": {},
                "consumer_protocol_ready": False,
            }, None

    def update(self, operation: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        for _ in range(16):
            value, token = self.read()
            operation(value)
            value["revision"] = uuid.uuid4().hex
            try:
                self.store.put(self.key, encode(value), expected=token)
                return value
            except ConflictError:
                # SDK retries can turn an acknowledged-lost success into a 412.
                committed, _ = self.read()
                if committed.get("revision") == value["revision"]:
                    return committed
                continue
            except Exception:
                # A timed-out PUT may already have committed. Never report failure
                # solely from the lost response, nor replay against a newer revision.
                committed, _ = self.read()
                if committed.get("revision") == value["revision"]:
                    return committed
                raise
        raise ConflictError("registry contention")

    def start(self, owner: str, fingerprint: str, *, force: bool = False) -> bool:
        def operation(value: dict[str, Any]) -> None:
            if value.get("active_run") not in {None, owner}:
                raise ConflictError("another run is active; explicit recovery required")
            if not force and any(
                r.get("fingerprint") == fingerprint and r.get("status") in {"published", "rejected"}
                for r in value["runs"].values()
            ):
                raise FileExistsError("identical published run")
            value["active_run"] = owner
            value["runs"].setdefault(
                owner,
                {
                    "fingerprint": fingerprint,
                    "status": "running",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )

        try:
            self.update(operation)
            return True
        except FileExistsError:
            return False

    def finish(self, owner: str, status: str) -> None:
        def operation(value: dict[str, Any]) -> None:
            if value.get("active_run") != owner:
                raise ConflictError("run ownership changed")
            was_published = any(
                entry["producer"] == owner and entry["published"]
                for entry in value["versions"].values()
            )
            value["runs"][owner]["status"] = "published" if was_published else status
            value["active_run"] = None

        self.update(operation)

    def register(self, version: str, metadata: dict[str, Any], owner: str) -> None:
        def operation(value: dict[str, Any]) -> None:
            if value.get("active_run") != owner:
                raise ConflictError("producer must own active run")
            run_id = metadata.get("run_id", owner)
            if run_id in value.get("retired_local_runs", []):
                raise ConflictError("training run has been retired locally")
            existing = value["versions"].get(version)
            if existing is not None:
                if (
                    existing["manifest_sha256"] != metadata["manifest_sha256"]
                    or existing["objects"] != metadata["objects"]
                ):
                    raise ConflictError("version collision")
                if existing["state"] != "uploading":
                    raise ConflictError("version already published or retired")
                if existing["producer"] != owner:
                    old_status = value["runs"][existing["producer"]]["status"]
                    if old_status not in {"failed", "aborted"}:
                        raise ConflictError("original producer has not terminated")
                    existing["producer"] = owner
                return
            value["versions"][version] = {
                **metadata,
                "state": "uploading",
                "pins": {},
                "producer": owner,
                "published": False,
            }

        self.update(operation)

    def publish(self, version: str, owner: str, *, rollback: bool = False) -> None:
        now = datetime.now(UTC).isoformat()

        def operation(value: dict[str, Any]) -> None:
            entry = value["versions"][version]
            if rollback:
                if not entry["published"] or entry["state"] not in {"available", "draining"}:
                    raise ValueError("version cannot be rolled back")
            elif entry.get("kind") == "diagnostic":
                raise ValueError("diagnostic run cannot be published")
            elif value.get("active_run") != owner or entry["producer"] != owner:
                raise ConflictError("producer ownership changed")
            if entry["state"] in {"deleting", "deleted"}:
                raise ValueError("version retired")
            current = value["current"]
            candidate = (entry["date"], instant(entry["created_at"]))
            high = value.get("publication_high_watermark")
            if high is None and current:
                previous = value["versions"][current]
                high = [previous["date"], previous["created_at"]]
            if not rollback and high and candidate < (high[0], instant(high[1])):
                raise ValueError("stale automatic publication")
            if not rollback:
                value["publication_high_watermark"] = [entry["date"], entry["created_at"]]
            if current and current != version:
                value["versions"][current]["replaced_at"] = now
            old = value["dates"].get(entry["date"])
            if old and old != version:
                value["versions"][old]["replaced_at"] = now
            entry.update(state="available", published=True, published_at=now)
            entry.pop("drain_started_at", None)
            value["current"] = version
            value["dates"][entry["date"]] = version

        self.update(operation)

    def pin(self, version: str, owner: str, *, release: bool = False) -> None:
        if not owner:
            raise ValueError("instance owner required")

        def operation(value: dict[str, Any]) -> None:
            entry = value["versions"][version]
            if release:
                entry["pins"].pop(owner, None)
            else:
                if entry["state"] != "available" or not entry["published"]:
                    raise ValueError("version not available for new references")
                entry["pins"].setdefault(owner, datetime.now(UTC).isoformat())

        self.update(operation)

    def consumer_ready(self) -> None:
        def operation(value: dict[str, Any]) -> None:
            value["consumer_protocol_ready"] = True

        self.update(operation)

    def cleanup(self, *, apply: bool = False, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("timezone required")
        plan: list[dict[str, str]] = []

        def operation(value: dict[str, Any]) -> None:
            plan.clear()
            for version, entry in value["versions"].items():
                reason = "retained"
                if version == value["current"]:
                    reason = "current"
                elif value.get("active_run") == entry["producer"]:
                    reason = "producer_active"
                elif entry["state"] == "deleted":
                    reason = "deleted"
                elif now - instant(entry["created_at"]) <= timedelta(days=30):
                    reason = "younger_than_30_days"
                elif entry.get("replaced_at") and now - instant(entry["replaced_at"]) < timedelta(
                    hours=24
                ):
                    reason = "replacement_grace"
                elif entry["state"] in {"available", "uploading"}:
                    reason = "drain"
                    entry.update(state="draining", drain_started_at=now.isoformat())
                    value["dates"] = {d: v for d, v in value["dates"].items() if v != version}
                elif entry["pins"]:
                    reason = "active_pins"
                elif now - instant(entry["drain_started_at"]) < timedelta(hours=24):
                    reason = "drain_grace"
                else:
                    reason = "delete"
                    entry["state"] = "deleting"
                plan.append({"version": version, "action": reason})

        value, _ = self.read()
        if not apply or not value["consumer_protocol_ready"]:
            operation(copy.deepcopy(value))
            return {
                "dry_run": True,
                "consumer_protocol_ready": value["consumer_protocol_ready"],
                "plan": plan,
            }
        value = self.update(operation)
        for step in plan:
            if step["action"] != "delete":
                continue
            version = step["version"]
            entry = self.read()[0]["versions"][version]
            if entry["state"] == "deleted":
                continue  # another cleaner already completed the same immutable version
            if entry["state"] != "deleting":
                raise ConflictError("deletion state changed")
            for key in entry["objects"]:
                if not key.startswith(self.name + "/" + entry["date"] + "/" + version + "/"):
                    raise ValueError("unowned deletion key")
                self.store.delete(key)

            def deleted(state: dict[str, Any], v: str = version) -> None:
                if state["versions"][v]["state"] not in {"deleting", "deleted"}:
                    raise ConflictError("deletion state changed")
                state["versions"][v]["state"] = "deleted"

            self.update(deleted)
        return {"dry_run": False, "plan": plan}


def verify_objects(store: ObjectStore, entry: dict[str, Any]) -> None:
    for key, expected in entry["objects"].items():
        if hashlib.sha256(store.get(key)[0]).hexdigest() != expected:
            raise ValueError("published object checksum mismatch")


def cleanup_local(
    registry: Registry, run_root: Path, export_root: Path, *, apply: bool
) -> list[str]:
    """Only remove local directories whose remote tombstone already fences all references."""
    import shutil

    from recommend.train.comm.run_lock import run_lock

    def unreferenced(state: dict[str, Any], run_id: str) -> bool:
        return state.get("active_run") != run_id and all(
            e["state"] == "deleted"
            for e in state["versions"].values()
            if e.get("run_id", e["producer"]) == run_id
        )

    state, _ = registry.read()
    removed: list[str] = []
    visited: set[str] = set()
    for version, entry in state["versions"].items():
        if entry["state"] != "deleted":
            continue
        relative = Path(registry.name) / entry["date"] / version
        export = (export_root / relative).resolve()
        if not export.is_relative_to(export_root.resolve()) or export == export_root.resolve():
            raise ValueError("invalid local cleanup path")
        if export.is_dir():
            removed.append(str(export))
            if apply:
                shutil.rmtree(export)
        run_id = entry.get("run_id", entry["producer"])
        path = (run_root / run_id).resolve()
        if not path.is_relative_to(run_root.resolve()) or path == run_root.resolve():
            raise ValueError("invalid local cleanup path")
        if run_id in visited or not path.is_dir() or not unreferenced(state, run_id):
            continue
        visited.add(run_id)
        if not apply:
            removed.append(str(path))
            continue
        try:
            with run_lock(path, exclusive=True):

                def retire(latest: dict[str, Any], rid: str = run_id) -> None:
                    if not unreferenced(latest, rid):
                        raise ConflictError("training run still referenced")
                    latest["retired_local_runs"] = sorted(
                        set(latest.get("retired_local_runs", [])) | {rid}
                    )

                registry.update(retire)
                shutil.rmtree(path)
                removed.append(str(path))
        except (BlockingIOError, ConflictError):
            continue  # active local reader or newly registered version: retain and retry later
    return removed
