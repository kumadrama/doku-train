"""Explicit file/S3 object storage with conditional writes and bounded reads."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


class ConflictError(RuntimeError):
    pass


class ObjectStore:
    def __init__(self, root: str, *, max_bytes: int = 268435456) -> None:
        self.root = root.rstrip("/")
        self.url = urlsplit(self.root)
        if self.url.scheme not in {"file", "s3"} or self.url.query or self.url.fragment:
            raise ValueError("explicit file or S3 root required")
        if self.url.scheme == "file" and self.url.netloc not in {"", "localhost"}:
            raise ValueError("invalid file host")
        if self.url.scheme == "s3" and (not self.url.netloc or not self.url.path.strip("/")):
            raise ValueError("S3 bucket and scoped prefix required")
        self.max_bytes = max_bytes
        self._client: Any = None

    def key(self, key: str) -> str:
        if (
            not key
            or key.startswith("/")
            or any(p in {"", ".", ".."} for p in unquote(key).split("/"))
            or "\\" in key
        ):
            raise ValueError("invalid object key")
        return self.url.path.lstrip("/") + "/" + key

    def path(self, key: str) -> Path:
        self.key(key)
        root = Path(unquote(self.url.path)).resolve()
        result = (root / key).resolve()
        if not result.is_relative_to(root):
            raise ValueError("object escapes root")
        return result

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.Session().client("s3")
        return self._client

    def get(self, key: str) -> tuple[bytes, str]:
        if self.url.scheme == "file":
            path = self.path(key)
            if path.stat().st_size > self.max_bytes:
                raise ValueError("object exceeds read budget")
            data = path.read_bytes()
            return data, hashlib.sha256(data).hexdigest()
        try:
            response = self.client.get_object(Bucket=self.url.netloc, Key=self.key(key))
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise FileNotFoundError(key) from exc
            raise
        stream = response["Body"]
        try:
            if response["ContentLength"] > self.max_bytes:
                raise ValueError("object exceeds read budget")
            data = stream.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise ValueError("object exceeds read budget")
            return data, str(response["ETag"])
        finally:
            stream.close()

    @contextmanager
    def _lock(self) -> Iterator[None]:
        import fcntl

        root = Path(unquote(self.url.path))
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".store.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def put(self, key: str, data: bytes, *, expected: str | None = None) -> None:
        """None means create-only; otherwise compare exact previous object token."""
        if len(data) > self.max_bytes:
            raise ValueError("object exceeds write budget")
        if self.url.scheme == "file":
            with self._lock():
                path = self.path(key)
                token = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                if token != expected:
                    raise ConflictError(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            return
        args = {"IfNoneMatch": "*"} if expected is None else {"IfMatch": expected}
        try:
            self.client.put_object(Bucket=self.url.netloc, Key=self.key(key), Body=data, **args)
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "412",
                "409",
            }:
                raise ConflictError(key) from exc
            raise

    def delete(self, key: str) -> None:
        if self.url.scheme == "file":
            self.path(key).unlink(missing_ok=True)
        else:
            self.client.delete_object(Bucket=self.url.netloc, Key=self.key(key))

    def list(self) -> list[str]:
        if self.url.scheme == "file":
            root = Path(unquote(self.url.path))
            return sorted(
                str(p.relative_to(root))
                for p in root.rglob("*")
                if p.is_file() and p.name != ".store.lock"
            )
        prefix = self.url.path.lstrip("/") + "/"
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.url.netloc, Prefix=prefix
        )
        return [o["Key"][len(prefix) :] for page in pages for o in page.get("Contents", [])]

    def immutable(self, key: str, data: bytes) -> None:
        try:
            self.put(key, data)
        except Exception:
            # Includes a successful PUT whose response was lost. Equality is the
            # only evidence that this immutable write can be considered complete.
            if self.get(key)[0] != data:
                raise
