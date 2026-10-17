"""Versioned daily test observations; no network client or metric quality gate."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

FORMAT = "doku-test-auc/v1"
FILENAME = "test-auc-events.json"


def record_events(
    root: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    completed_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Freeze completion time once; replay the exact same observation on retry."""
    path = root / FILENAME
    previous = json.loads(path.read_text()) if path.exists() else None
    now = completed_at or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("completion time must have a timezone")
    stamp = int(now.timestamp() * 1000)
    if previous:
        stamp = previous[0]["completed_at_ms"]
    local = datetime.fromtimestamp(stamp / 1000, ZoneInfo("Asia/Shanghai"))
    day = int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    run_id = config["output"]["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("invalid observation run_id")
    events = []
    for task, options in sorted(config["tasks"].items()):
        for metric, definition in sorted(options["metrics"].items()):
            kind = definition["func"]
            if kind not in {"auc", "regression_auc"}:
                continue
            value = metrics[task]["metrics"][metric]["value"]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("invalid test AUC")
            events.append(
                {
                    "format": FORMAT,
                    "model_name": config["model_name"],
                    "task": task,
                    "metric": metric,
                    "kind": kind,
                    "run_id": run_id,
                    "completed_at_ms": stamp,
                    "day_start_ms": day,
                    "auc": value,
                    "observation": (
                        f"{stamp:020d}|{run_id}|{json.dumps({'auc': value}, allow_nan=False)}"
                    ),
                }
            )
    if previous is not None:
        if previous != events:
            raise ValueError("frozen test AUC observation changed")
        return events
    # Publish a complete file atomically, without overwriting a concurrent writer.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(events, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return events


def emit_events(events: list[dict[str, Any]], *, stream: TextIO | None = None) -> None:
    """Write JSON lines to the container's existing awslogs transport."""
    for event in events:
        print(
            json.dumps(event, sort_keys=True, allow_nan=False),
            file=stream or sys.stdout,
            flush=True,
        )
