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
SHARED_OUTPUT_NAMES = (
    "metrics.json",
    "fitted-feature-state/state.json",
    "lineage.json",
)


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


@dataclass(frozen=True, slots=True)
class ModelTrainingOutputUris:
    model_artifact: str
    metrics: str
    fitted_feature_state: str
    lineage: str


def _validate_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("invalid run_id")


def _normalize_json(name: str, body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON training output: {name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON training output must be an object: {name}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_files(
    store: ObjectStore,
    output_prefix: str,
    run_id: str,
    files: Mapping[str, bytes],
) -> dict[str, str]:
    uris = {name: f"{output_prefix.rstrip('/')}/{run_id}/{name}" for name in files}
    for name, body in files.items():
        uri = uris[name]
        try:
            store.put_bytes_if_absent(uri, body)
            readback = store.get_bytes(uri, max_bytes=len(body))
        except FileExistsError as error:
            raise TrainingOutputConflict(f"training output already exists: {name}") from error
        except Exception as error:
            raise TrainingOutputWriteError(f"failed to write training output: {name}") from error
        if hashlib.sha256(readback).digest() != hashlib.sha256(body).digest():
            raise TrainingOutputWriteError(f"training output readback mismatch: {name}")
    return uris


class TrainingOutputWriter:
    def __init__(self, store: ObjectStore, output_prefix: str) -> None:
        self._store = store
        self._output_prefix = output_prefix.rstrip("/")

    def write(self, run_id: str, files: Mapping[str, bytes]) -> TrainingOutputUris:
        _validate_run_id(run_id)
        if set(files) != set(TRAINING_OUTPUT_NAMES):
            raise ValueError("training output requires exactly four canonical files")
        normalized = {
            name: _normalize_json(name, files[name]) if name in JSON_OUTPUT_NAMES else files[name]
            for name in TRAINING_OUTPUT_NAMES
        }
        if not normalized["checkpoint.pt"]:
            raise ValueError("checkpoint.pt must not be empty")
        uris = _write_files(self._store, self._output_prefix, run_id, normalized)
        return TrainingOutputUris(
            checkpoint=uris["checkpoint.pt"],
            metrics=uris["metrics.json"],
            fitted_feature_state=uris["fitted-feature-state/state.json"],
            lineage=uris["lineage.json"],
        )


class ModelTrainingOutputWriter:
    def __init__(self, store: ObjectStore, output_prefix: str, artifact_name: str) -> None:
        if not artifact_name or "/" in artifact_name or artifact_name in SHARED_OUTPUT_NAMES:
            raise ValueError("invalid model artifact name")
        self._store = store
        self._output_prefix = output_prefix.rstrip("/")
        self._artifact_name = artifact_name

    def write(
        self,
        run_id: str,
        *,
        model_artifact: bytes,
        metrics: bytes,
        fitted_feature_state: bytes,
        lineage: bytes,
    ) -> ModelTrainingOutputUris:
        _validate_run_id(run_id)
        if not model_artifact:
            raise ValueError("model artifact must not be empty")
        files = {
            self._artifact_name: model_artifact,
            "metrics.json": _normalize_json("metrics.json", metrics),
            "fitted-feature-state/state.json": _normalize_json(
                "fitted-feature-state/state.json", fitted_feature_state
            ),
            "lineage.json": _normalize_json("lineage.json", lineage),
        }
        uris = _write_files(self._store, self._output_prefix, run_id, files)
        return ModelTrainingOutputUris(
            model_artifact=uris[self._artifact_name],
            metrics=uris["metrics.json"],
            fitted_feature_state=uris["fitted-feature-state/state.json"],
            lineage=uris["lineage.json"],
        )
