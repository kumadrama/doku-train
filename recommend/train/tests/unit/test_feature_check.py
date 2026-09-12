from __future__ import annotations

import pyarrow as pa
import pytest

from recommend.train.comm.feature_check import validate_feature_names, validate_point_in_time

DAY_MS = 86_400_000


def test_raw_user_id_is_rejected_from_model_features() -> None:
    with pytest.raises(ValueError, match="user_id"):
        validate_feature_names(("watch_7d", "user_id"))


def test_point_in_time_requires_exact_past_seven_day_window() -> None:
    event_time = 1_800_000_000_000
    valid = pa.record_batch(
        {
            "event_time_ms": [event_time],
            "watch_7d__window_start_ms": [event_time - 7 * DAY_MS],
            "watch_7d__window_end_ms": [event_time],
        }
    )
    validate_point_in_time(valid, ("watch_7d",))

    future = valid.set_column(
        2, "watch_7d__window_end_ms", pa.array([event_time + 1], type=pa.int64())
    )
    with pytest.raises(ValueError, match="FEATURE_TIME_LEAKAGE"):
        validate_point_in_time(future, ("watch_7d",))

    wrong_window = valid.set_column(
        1,
        "watch_7d__window_start_ms",
        pa.array([event_time - 6 * DAY_MS], type=pa.int64()),
    )
    with pytest.raises(ValueError, match="FEATURE_WINDOW_INVALID"):
        validate_point_in_time(wrong_window, ("watch_7d",))
