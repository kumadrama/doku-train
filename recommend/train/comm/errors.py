from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    run_id: str
    category: str
    stage: str
    message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def classify_error(error: Exception, *, run_id: str, stage: str) -> ErrorEnvelope:
    raw = str(error)
    category = "TRAINING_FAILED"
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
    if isinstance(error, (FileNotFoundError, ValueError)) and category == "TRAINING_FAILED":
        category = "INPUT_CONTRACT_INVALID"
    redacted = re.sub(r"(?i)(token|password|secret|signature)=[^&\s]+", r"\1=<redacted>", raw)
    redacted = redacted.split("?", maxsplit=1)[0]
    return ErrorEnvelope(run_id=run_id, category=category, stage=stage, message=redacted)
