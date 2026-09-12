from __future__ import annotations

import platform
import resource
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class ResourceReport:
    cpu_model: str
    cpu_count: int
    total_memory_bytes: int
    available_memory_bytes: int
    peak_rss_bytes: int
    row_count: int
    batch_count: int
    byte_count: int
    wall_seconds: float
    rows_per_second: float
    batches_per_second: float
    bytes_per_second: float
    stage_seconds: Mapping[str, float]


class ResourceTracker:
    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._stage_seconds: dict[str, float] = {}
        self._peak_rss_bytes = self._current_peak_rss_bytes()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not name:
            raise ValueError("resource stage name must not be empty")
        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started_at
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed
            self._peak_rss_bytes = max(
                self._peak_rss_bytes,
                self._current_peak_rss_bytes(),
            )

    def report(
        self,
        *,
        row_count: int,
        batch_count: int,
        byte_count: int,
    ) -> ResourceReport:
        if min(row_count, batch_count, byte_count) < 0:
            raise ValueError("resource volumes must be non-negative")
        wall_seconds = time.perf_counter() - self._started_at
        rate_window = max(wall_seconds, 1e-12)
        memory = psutil.virtual_memory()
        peak_rss_bytes = max(self._peak_rss_bytes, self._current_peak_rss_bytes())
        return ResourceReport(
            cpu_model=platform.processor() or platform.machine() or "unknown",
            cpu_count=psutil.cpu_count(logical=True) or 1,
            total_memory_bytes=memory.total,
            available_memory_bytes=memory.available,
            peak_rss_bytes=peak_rss_bytes,
            row_count=row_count,
            batch_count=batch_count,
            byte_count=byte_count,
            wall_seconds=wall_seconds,
            rows_per_second=row_count / rate_window,
            batches_per_second=batch_count / rate_window,
            bytes_per_second=byte_count / rate_window,
            stage_seconds=dict(sorted(self._stage_seconds.items())),
        )

    @staticmethod
    def _current_peak_rss_bytes() -> int:
        current = int(psutil.Process().memory_info().rss)
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            maximum *= 1024
        return max(current, maximum)
