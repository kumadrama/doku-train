from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from recommend.train.comm.training_output import (
    TrainingOutputConflict,
    TrainingOutputWriteError,
)


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    run_id: str
    category: str
    stage: str
    message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class StageFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(str(cause))


@contextmanager
def error_stage(stage: str) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        if isinstance(error, StageFailure) or getattr(error, "stage", None) is not None:
            raise
        raise StageFailure(stage, error) from error


def classify_error(error: Exception, *, run_id: str, stage: str) -> ErrorEnvelope:
    root_error = error.cause if isinstance(error, StageFailure) else error
    error_stage_name = getattr(error, "stage", None)
    effective_stage = error_stage_name if isinstance(error_stage_name, str) else stage
    raw = str(root_error)
    category = "TRAINING_FAILED"
    if isinstance(root_error, TrainingOutputConflict):
        category = "TRAINING_OUTPUT_CONFLICT"
    elif isinstance(root_error, TrainingOutputWriteError):
        category = "TRAINING_OUTPUT_WRITE_FAILED"
    for candidate in (
        "INPUT_CONTRACT_INVALID",
        "DATA_INTEGRITY_MISMATCH",
        "LABEL_NOT_MATURE",
        "SPLIT_INVALID",
        "RESOURCE_BUDGET_EXCEEDED",
        "NUMERICAL_FAILURE",
        "CHECKPOINT_INCOMPATIBLE",
        "EVALUATION_GATE_FAILED",
        "TRAINING_OUTPUT_CONFLICT",
        "TRAINING_OUTPUT_WRITE_FAILED",
    ):
        if candidate in raw:
            category = candidate
            break
    if isinstance(root_error, (FileNotFoundError, ValueError)) and category == "TRAINING_FAILED":
        category = "INPUT_CONTRACT_INVALID"
    redacted = re.sub(r"(?i)(token|password|secret|signature)=[^&\s]+", r"\1=<redacted>", raw)
    redacted = redacted.split("?", maxsplit=1)[0]
    return ErrorEnvelope(
        run_id=run_id,
        category=category,
        stage=effective_stage,
        message=redacted,
    )
