from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.training_output import (
    TRAINING_OUTPUT_NAMES,
    ModelTrainingOutputWriter,
    TrainingOutputConflict,
    TrainingOutputWriteError,
    TrainingOutputWriter,
)


def files() -> dict[str, bytes]:
    return {
        "checkpoint.pt": b"checkpoint",
        "metrics.json": b'{"z":1,"a":2}',
        "fitted-feature-state/state.json": b'{"state_version":"1"}',
        "lineage.json": b'{"run_id":"run-1"}',
    }


def test_writer_commits_exact_four_create_only_outputs(tmp_path: Path) -> None:
    writer = TrainingOutputWriter(LocalStorage(tmp_path), tmp_path.as_uri())
    uris = writer.write("run-1", files())
    assert set(TRAINING_OUTPUT_NAMES) == set(files())
    assert uris.checkpoint.endswith("/run-1/checkpoint.pt")
    assert uris.metrics.endswith("/run-1/metrics.json")
    assert uris.fitted_feature_state.endswith("/run-1/fitted-feature-state/state.json")
    assert uris.lineage.endswith("/run-1/lineage.json")
    assert json.loads((tmp_path / "run-1" / "metrics.json").read_bytes()) == {"a": 2, "z": 1}
    assert sorted(
        path.relative_to(tmp_path / "run-1").as_posix()
        for path in (tmp_path / "run-1").rglob("*")
        if path.is_file()
    ) == sorted(TRAINING_OUTPUT_NAMES)


def test_writer_rejects_incomplete_or_extra_file_sets(tmp_path: Path) -> None:
    writer = TrainingOutputWriter(LocalStorage(tmp_path), tmp_path.as_uri())
    incomplete = files()
    incomplete.pop("lineage.json")
    with pytest.raises(ValueError, match="exactly four"):
        writer.write("run-incomplete", incomplete)
    with pytest.raises(ValueError, match="exactly four"):
        writer.write("run-extra", files() | {"manifest.json": b"{}"})


def test_writer_never_overwrites_existing_output(tmp_path: Path) -> None:
    writer = TrainingOutputWriter(LocalStorage(tmp_path), tmp_path.as_uri())
    writer.write("run-1", files())
    with pytest.raises(TrainingOutputConflict):
        writer.write("run-1", files() | {"checkpoint.pt": b"replacement"})
    assert (tmp_path / "run-1" / "checkpoint.pt").read_bytes() == b"checkpoint"


class FailingStore(LocalStorage):
    def put_bytes_if_absent(self, uri: str, body: bytes):  # type: ignore[no-untyped-def]
        if uri.endswith("lineage.json"):
            raise OSError("injected failure")
        return super().put_bytes_if_absent(uri, body)


def test_partial_write_is_not_reported_as_success(tmp_path: Path) -> None:
    writer = TrainingOutputWriter(FailingStore(tmp_path), tmp_path.as_uri())
    with pytest.raises(TrainingOutputWriteError, match=r"lineage\.json"):
        writer.write("run-1", files())
    assert not (tmp_path / "run-1" / "lineage.json").exists()


def test_model_writer_supports_xgboost_artifact_without_weakening_shared_outputs(
    tmp_path: Path,
) -> None:
    writer = ModelTrainingOutputWriter(
        LocalStorage(tmp_path),
        tmp_path.as_uri(),
        "checkpoint.xgb.zip",
    )
    uris = writer.write(
        "xgboost-run",
        model_artifact=b"xgboost-checkpoint",
        metrics=b'{"metric_set_version":"1"}',
        fitted_feature_state=b'{"state_version":"1"}',
        lineage=b'{"model_name":"rerank_tree"}',
    )
    assert uris.model_artifact.endswith("/xgboost-run/checkpoint.xgb.zip")
    assert sorted(
        path.relative_to(tmp_path / "xgboost-run").as_posix()
        for path in (tmp_path / "xgboost-run").rglob("*")
        if path.is_file()
    ) == sorted(
        [
            "checkpoint.xgb.zip",
            "metrics.json",
            "fitted-feature-state/state.json",
            "lineage.json",
        ]
    )
