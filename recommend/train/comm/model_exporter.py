"""Future post-refit model delivery boundary.

Spec 001 deliberately provides no implementation or composition-root registration. A future
implementation receives only successful training outputs and owns an independent attempt status;
its failure cannot change the completed refit or training-run state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from recommend.train.comm.training_output import TrainingOutputUris


@dataclass(frozen=True, slots=True)
class ExportRequest:
    run_id: str
    training_outputs: TrainingOutputUris


@dataclass(frozen=True, slots=True)
class ExportResult:
    export_attempt_id: str
    status: Literal["SUCCEEDED", "FAILED"]
    artifact_uri: str | None
    error_category: str | None


@runtime_checkable
class ModelExporter(Protocol):
    def export(self, request: ExportRequest) -> ExportResult: ...
