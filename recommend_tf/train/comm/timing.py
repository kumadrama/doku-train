"""Opt-in elapsed-time diagnostics; independent of model and training RNG."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
_path: ContextVar[tuple[str, ...]] = ContextVar("timing_path", default=())


class Timer:
    def __init__(self, root: Path | None = None, phase: str = "train", print_logs: bool = False):
        self.enabled = root is not None
        self.phase = phase
        self.print_logs = print_logs
        self.totals: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.epoch_previous: dict[str, float] = {}
        if root is not None:
            directory = root / "timing"
            directory.mkdir(parents=True, exist_ok=True)
            self.invocation_id = uuid.uuid4().hex
            stem = f"{phase}-{self.invocation_id}"
            self.summary = directory / f"{stem}-summary.json"
            self.stream = (directory / f"{stem}.jsonl").open("x")
            self.run_id = root.name
            self.started = self.last_progress = time.perf_counter()

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        with self.lock:
            payload = {
                "version": "doku-timing/v1",
                "run_id": self.run_id,
                "phase": self.phase,
                "invocation_id": self.invocation_id,
                "event": event,
                **fields,
            }
            line = json.dumps(payload, sort_keys=True, allow_nan=False)
            self.stream.write(line + "\n")
            self.stream.flush()
            if self.print_logs:
                print("DOKU_TIMING " + line, flush=True)

    def record(self, name: str, elapsed: float, count: int = 1, unit: str = "calls") -> None:
        with self.lock:
            item = self.totals.setdefault(
                name,
                {
                    "elapsed_seconds": 0.0,
                    "count": 0,
                    "unit": unit,
                    "semantics": "inclusive_elapsed; worker durations may overlap",
                },
            )
            item["elapsed_seconds"] += elapsed
            item["count"] += count
            item["units_per_second"] = (
                item["count"] / item["elapsed_seconds"] if item["elapsed_seconds"] else None
            )

    def progress(self) -> None:
        now = time.perf_counter()
        with self.lock:
            if now - self.last_progress >= 60:
                self.last_progress = now
                self.emit("progress", elapsed_seconds=now - self.started, stages=self.totals)

    def span(self, name: str, *, report: bool = False, count: int = 1, unit: str = "calls") -> Any:
        if not self.enabled:
            return nullcontext()
        return self._span(name, report, count, unit)

    @contextmanager
    def _span(self, name: str, report: bool, count: int, unit: str) -> Iterator[None]:
        path = (*_path.get(), name)
        key = "/".join(path)
        token = _path.set(path)
        started = time.perf_counter()
        status = "complete"
        if report:
            self.emit("stage_start", stage=key)
        try:
            yield
        except BaseException as exc:
            status = "failed"
            self.emit("stage_failed", stage=key, error_type=type(exc).__name__)
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.record(key, elapsed, count, unit)
            _path.reset(token)
            if report:
                self.emit("stage_end", stage=key, status=status, elapsed_seconds=elapsed)
            self.progress()

    def iterate(self, name: str, values: Iterator[T], *, unit: str = "batches") -> Iterator[T]:
        if not self.enabled:
            yield from values
            return
        while True:
            started = time.perf_counter()
            try:
                value = next(values)
            except StopIteration:
                return
            except BaseException as exc:
                self.record(
                    "/".join((*_path.get(), name)),
                    time.perf_counter() - started,
                    count=0,
                    unit=unit,
                )
                self.emit(
                    "stage_failed",
                    stage="/".join((*_path.get(), name)),
                    error_type=type(exc).__name__,
                )
                raise
            self.record("/".join((*_path.get(), name)), time.perf_counter() - started, unit=unit)
            self.progress()
            yield value  # Never include consumer time in an iterator measurement.

    def epoch(self, epoch: int) -> None:
        if self.enabled:
            delta = {
                k: v["elapsed_seconds"] - self.epoch_previous.get(k, 0.0)
                for k, v in self.totals.items()
            }
            self.emit("epoch_summary", epoch=epoch, stages=self.totals, delta_seconds=delta)
            self.epoch_previous = {k: v["elapsed_seconds"] for k, v in self.totals.items()}

    def close(self, status: str) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        try:
            self.emit("session_end", status=status, elapsed_seconds=elapsed, stages=self.totals)
            partial = self.summary.with_suffix(".partial")
            partial.write_text(
                json.dumps(
                    {
                        "version": "doku-timing/v1",
                        "run_id": self.run_id,
                        "phase": self.phase,
                        "invocation_id": self.invocation_id,
                        "status": status,
                        "elapsed_seconds": elapsed,
                        "stages": self.totals,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            partial.replace(self.summary)
        finally:
            self.stream.close()


_OFF = Timer()
_current: ContextVar[Timer] = ContextVar("timing_collector", default=_OFF)


def current() -> Timer:
    return _current.get()


@contextmanager
def session(config: dict[str, Any], root: Path, phase: str) -> Iterator[Timer]:
    options = config.get("profiling", {})
    timer = Timer(root, phase, options.get("print_logs", False)) if options.get("enabled") else _OFF
    token = _current.set(timer)
    path_token = _path.set(())
    status = "complete"
    try:
        yield timer
    except BaseException:
        status = "failed"
        raise
    finally:
        try:
            timer.close(status)
        finally:
            _current.reset(token)
            _path.reset(path_token)


def measured(name: str, *, unit: str = "calls") -> Callable[..., Any]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            timer = current()
            if not timer.enabled:
                return function(*args, **kwargs)
            with timer.span(name, unit=unit):
                return function(*args, **kwargs)

        return call

    return decorate
