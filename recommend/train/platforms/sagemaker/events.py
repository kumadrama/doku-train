from __future__ import annotations

import json
import math
import threading
import uuid
from pathlib import Path
from typing import Any

from recommend.train.platforms.sagemaker.submit import put_once


class EventSink:
    """Rotate closed event files into immutable S3 objects; background sync is observable."""

    def __init__(self, directory: Path, s3: Any, uri: str, key: str, interval: int) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.s3, self.uri, self.key = s3, uri, key
        self.lock = threading.Lock()
        self.pending: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self.thread.start()

    def __call__(self, event: dict[str, Any]) -> None:
        if self.error:
            raise RuntimeError("metric upload failed") from self.error
        print("DOKU_METRIC " + json.dumps(event, allow_nan=False), flush=True)
        value = event.get("value")
        if value is not None and math.isfinite(float(value)):
            name = f"{event['split']}/{event['task']}/{event['metric']}"
            print(f"SM_METRIC {name}={value}", flush=True)
            with self.lock:
                self.pending.append(dict(event))

    def flush(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        with self.lock:
            if not self.pending:
                return
            pending, self.pending = self.pending, []
        directory = self.directory / uuid.uuid4().hex
        with SummaryWriter(str(directory)) as writer:
            for e in pending:
                writer.add_scalar(
                    f"{e['split']}/{e['task']}/{e['metric']}", float(e["value"]), int(e["step"])
                )
        for path in directory.iterdir():
            put_once(self.s3, self.uri + path.name, path.read_bytes(), self.key)

    def _loop(self, interval: int) -> None:
        while not self.stopped.wait(interval):
            try:
                self.flush()
            except Exception as exc:
                self.error = exc
                return

    def close(self) -> None:
        self.stopped.set()
        self.thread.join()
        if self.error:
            raise RuntimeError("metric upload failed") from self.error
        self.flush()
