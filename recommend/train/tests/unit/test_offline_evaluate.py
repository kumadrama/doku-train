from __future__ import annotations

from datetime import date, timedelta

from recommend.train.comm.offline_evaluate import build_backtest_windows


def test_24_2_2_backtest_is_explicit_and_non_overlapping() -> None:
    dates = tuple(date(2026, 1, 1) + timedelta(days=offset) for offset in range(28))
    windows = build_backtest_windows(dates, train_days=24, validation_days=2, test_days=2)
    assert len(windows) == 1
    assert len(windows[0].train) == 24
    assert len(windows[0].validation) == 2
    assert len(windows[0].test) == 2
    assert set(windows[0].train).isdisjoint(windows[0].validation + windows[0].test)
