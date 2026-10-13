"""Coordinate local evaluation/export readers with cleanup, outside removable run dirs."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_lock(run_dir: Path, *, exclusive: bool = False) -> Iterator[None]:
    root = run_dir.resolve()
    locks = root.parent / ".run-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (root.name + ".lock")).open("a") as handle:
        flags = (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle, flags)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
