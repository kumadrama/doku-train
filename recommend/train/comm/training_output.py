from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from recommend.train.comm.datasvr.storage import ObjectStore

TRAINING_OUTPUT_NAMES = (
    "checkpoint.pt",
    "metrics.json",
    "fitted-feature-state/state.json",
    "lineage.json",
)
JSON_OUTPUT_NAMES = frozenset(TRAINING_OUTPUT_NAMES[1:])


class TrainingOutputConflict(RuntimeError):
    pass


class TrainingOutputWriteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingOutputUris:
    checkpoint: str
    metrics: str
    fitted_feature_state: str
    lineage: str


class TrainingOutputWriter:
    def __init__(self, store: ObjectStore, output_prefix: str) -> None:
        self._store = store
        self._output_prefix = output_prefix.rstrip("/")

    def write(self, run_id: str, files: Mapping[str, bytes]) -> TrainingOutputUris:
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("invalid run_id")
        if set(files) != set(TRAINING_OUTPUT_NAMES):
            raise ValueError("training output requires exactly four canonical files")
        normalized = {
            name: self._normalize_json(name, files[name])
            if name in JSON_OUTPUT_NAMES
            else files[name]
            for name in TRAINING_OUTPUT_NAMES
        }
        if not normalized["checkpoint.pt"]:
            raise ValueError("checkpoint.pt must not be empty")
        uris = {name: f"{self._output_prefix}/{run_id}/{name}" for name in TRAINING_OUTPUT_NAMES}
        for name in TRAINING_OUTPUT_NAMES:
            body = normalized[name]
            uri = uris[name]
            try:
                self._store.put_bytes_if_absent(uri, body)
                readback = self._store.get_bytes(uri, max_bytes=len(body))
            except FileExistsError as error:
                raise TrainingOutputConflict(f"training output already exists: {name}") from error
            except Exception as error:
                raise TrainingOutputWriteError(
                    f"failed to write training output: {name}"
                ) from error
            if hashlib.sha256(readback).digest() != hashlib.sha256(body).digest():
                raise TrainingOutputWriteError(f"training output readback mismatch: {name}")
        return TrainingOutputUris(
            checkpoint=uris["checkpoint.pt"],
            metrics=uris["metrics.json"],
            fitted_feature_state=uris["fitted-feature-state/state.json"],
            lineage=uris["lineage.json"],
        )

    @staticmethod
    def _normalize_json(name: str, body: bytes) -> bytes:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON training output: {name}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"JSON training output must be an object: {name}")
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
