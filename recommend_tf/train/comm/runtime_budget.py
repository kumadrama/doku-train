"""Per-process run memory sampling, independent of native model libraries."""

from __future__ import annotations

import threading

import psutil

_active: MemoryBudget | None = None


class MemoryBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
            self.stop_event.wait(0.05)

    def start(self) -> None:
        global _active
        _active = self
        self.thread.start()
        self.check()

    def check(self, additional: int = 0) -> None:
        rss = psutil.Process().memory_info().rss
        self.peak = max(self.peak, rss)
        if max(self.peak, rss + additional) > self.limit:
            raise MemoryError("RESOURCE_BUDGET_EXCEEDED: process memory limit")

    def close(self) -> None:
        global _active
        self.stop_event.set()
        self.thread.join()
        _active = None


def check_memory(additional: int = 0) -> None:
    if _active is not None:
        _active.check(additional)
