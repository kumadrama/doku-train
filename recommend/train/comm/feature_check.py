from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

SEVEN_DAYS_MS = 7 * 86_400_000


def validate_feature_names(feature_names: Iterable[str]) -> None:
    forbidden = {name for name in feature_names if name.casefold() == "user_id"}
    if forbidden:
        raise ValueError("raw user_id is forbidden from the model feature signature")


def validate_point_in_time(batch: pa.RecordBatch, statistical_features: Iterable[str]) -> None:
    event_times = batch.column("event_time_ms").to_pylist()
    for feature in statistical_features:
        starts = batch.column(f"{feature}__window_start_ms").to_pylist()
        ends = batch.column(f"{feature}__window_end_ms").to_pylist()
        for event_time, start, end in zip(event_times, starts, ends, strict=True):
            if event_time is None or start is None or end is None:
                raise ValueError(f"FEATURE_WINDOW_INVALID: missing window metadata for {feature}")
            if int(end) > int(event_time):
                raise ValueError(f"FEATURE_TIME_LEAKAGE: {feature} window ends after exposure")
            if int(end) - int(start) != SEVEN_DAYS_MS:
                raise ValueError(f"FEATURE_WINDOW_INVALID: {feature} is not a seven-day window")
