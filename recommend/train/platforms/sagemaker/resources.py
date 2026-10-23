"""Sample local volume use; values are sampled peaks, not a reservation or SLA."""

from __future__ import annotations

import resource
import shutil
import sys
import threading
from pathlib import Path
from typing import Any


class ResourceSampler:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.stop = threading.Event()
        self.peaks: dict[str, int] = {}
        self.min_free = shutil.disk_usage(root).free
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def sample(self) -> None:
        self.min_free = min(self.min_free, shutil.disk_usage(self.root).free)
        for name in ("input", "work", "model", "output"):
            total = 0
            for path in (self.root / name).rglob("*"):
                try:
                    if path.is_file():
                        total += path.stat().st_size
                except FileNotFoundError:
                    pass  # Temporary files can disappear during sampling.
            self.peaks[name] = max(self.peaks.get(name, 0), total)

    def _run(self) -> None:
        while not self.stop.wait(5):
            self.sample()

    def close(self) -> dict[str, Any]:
        self.stop.set()
        self.thread.join()
        self.sample()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "sample_interval_seconds": 5,
            "sampled_directory_peak_bytes": self.peaks,
            "sampled_min_disk_free_bytes": self.min_free,
            "peak_rss_bytes": int(rss * (1 if sys.platform == "darwin" else 1024)),
        }
