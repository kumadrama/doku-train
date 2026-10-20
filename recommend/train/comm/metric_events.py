"""Optional, process-local metric observer; no platform dependency or training semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Observer = Callable[[dict[str, Any]], None]
_observer: ContextVar[Observer | None] = ContextVar("metric_observer", default=None)
_interval: ContextVar[int] = ContextVar("metric_interval", default=100)


@contextmanager
def observe(observer: Observer, interval: int = 100) -> Iterator[None]:
    if interval < 1:
        raise ValueError("metric interval must be positive")
    token, cadence = _observer.set(observer), _interval.set(interval)
    try:
        yield
    finally:
        _observer.reset(token)
        _interval.reset(cadence)


def emit(event: dict[str, Any]) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(event)


def training_step(step: int, epoch: int, losses: dict[str, float | None]) -> None:
    if step % _interval.get() == 0:
        for task, value in losses.items():
            emit(dict(split="train", task=task, metric="loss", step=step, epoch=epoch, value=value))


def evaluation(split: str, step: int, epoch: int, results: dict[str, Any]) -> None:
    for task, result in results.items():
        emit(
            dict(
                split=split, task=task, metric="loss", step=step, epoch=epoch, value=result["loss"]
            )
        )
        for name, metric in result["metrics"].items():
            if isinstance(metric, dict):
                emit(dict(split=split, task=task, metric=name, step=step, epoch=epoch, **metric))
