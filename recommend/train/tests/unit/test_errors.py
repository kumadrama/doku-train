from __future__ import annotations

import pytest

from recommend.train.comm.errors import classify_error
from recommend.train.comm.training_output import (
    TrainingOutputConflict,
    TrainingOutputWriteError,
)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (TrainingOutputConflict("already exists"), "TRAINING_OUTPUT_CONFLICT"),
        (TrainingOutputWriteError("readback mismatch"), "TRAINING_OUTPUT_WRITE_FAILED"),
    ],
)
def test_training_output_errors_have_stable_categories(error: Exception, category: str) -> None:
    envelope = classify_error(error, run_id="run-1", stage="write-output")
    assert envelope.category == category
