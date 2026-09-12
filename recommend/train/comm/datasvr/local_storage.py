from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO

from recommend.train.comm.datasvr.storage import ObjectHead, parse_uri


class LocalStorage:
    def __init__(self, allowed_root: Path) -> None:
        self._allowed_root = allowed_root.resolve()

    def _path(self, uri: str) -> Path:
        parsed = parse_uri(uri)
        if parsed.scheme != "file":
            raise ValueError("LocalStorage requires a file URI")
        path = Path(parsed.key).resolve()
        if not path.is_relative_to(self._allowed_root):
            raise ValueError("object is outside allowed root")
        return path

    @staticmethod
    def _head(path: Path) -> ObjectHead:
        body = path.read_bytes()
        return ObjectHead(len(body), hashlib.sha256(body).hexdigest())

    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        path = self._path(uri)
        with path.open("rb") as source:
            body = source.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("object exceeds max_bytes")
        return body

    def head(self, uri: str) -> ObjectHead:
        return self._head(self._path(uri))

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead:
        path = self._path(uri)
        with path.open("rb") as source:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        return self._head(path)

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(body)
        return ObjectHead(len(body), hashlib.sha256(body).hexdigest())
