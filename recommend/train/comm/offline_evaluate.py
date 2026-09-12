from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class BacktestWindow:
    train: tuple[date, ...]
    validation: tuple[date, ...]
    test: tuple[date, ...]


def build_backtest_windows(
    dates: Sequence[date],
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
) -> tuple[BacktestWindow, ...]:
    ordered = tuple(sorted(dates))
    if len(set(ordered)) != len(ordered):
        raise ValueError("SPLIT_INVALID: backtest dates must be distinct")
    if any(current != previous + timedelta(days=1) for previous, current in pairwise(ordered)):
        raise ValueError("SPLIT_INVALID: backtest dates must be consecutive")
    width = train_days + validation_days + test_days
    if min(train_days, validation_days, test_days) < 1 or len(ordered) < width:
        raise ValueError("SPLIT_INVALID: insufficient dates for backtest")
    windows: list[BacktestWindow] = []
    for start in range(0, len(ordered) - width + 1, test_days):
        train_end = start + train_days
        validation_end = train_end + validation_days
        windows.append(
            BacktestWindow(
                train=ordered[start:train_end],
                validation=ordered[train_end:validation_end],
                test=ordered[validation_end : validation_end + test_days],
            )
        )
    return tuple(windows)
