"""Binary Redis transport. No TTLs, implicit environment, or model selection."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from recommend.train.comm.artifact_store import ConflictError

# Only small metadata is inspected in Lua. Model bytes are uploaded outside it.
COMMIT_CATALOG = """
local actual = redis.call('GET', KEYS[1])
if actual ~= ARGV[1] and not (not actual and ARGV[1] == '') then
  if actual ~= ARGV[2] then return 0 end
end
for i = 2, #KEYS do
  if redis.call('GET', KEYS[i]) ~= ARGV[i + 1] then return -1 end
end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""


class RedisTransport(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def immutable(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> None: ...
    def commit(
        self, key: str, expected: bytes | None, value: bytes, ready: dict[str, bytes]
    ) -> None: ...


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid distribution identifier")
    return value


class RedisArtifactStore:
    def __init__(self, client: Any, *, max_file_bytes: int) -> None:
        if max_file_bytes < 1:
            raise ValueError("positive Redis file budget required")
        self.client = client
        self.max_file_bytes = max_file_bytes

    @classmethod
    def connect(
        cls, url: str, *, cluster: bool, max_file_bytes: int, timeout_seconds: float
    ) -> RedisArtifactStore:
        from redis import Redis
        from redis.cluster import RedisCluster

        parsed = urlsplit(url)
        # No URL query overrides: they could disable certificate checks or alter binary decoding.
        if (
            parsed.query
            or parsed.fragment
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("invalid Redis connection options")
        if parsed.scheme != "rediss" and not (
            parsed.scheme == "redis" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Redis requires TLS except for loopback tests")
        factory: Any = RedisCluster if cluster else Redis
        options: dict[str, Any] = {
            "decode_responses": False,
            "socket_timeout": timeout_seconds,
            "socket_connect_timeout": timeout_seconds,
        }
        if parsed.scheme == "rediss":
            options.update(ssl_cert_reqs="required", ssl_check_hostname=True)
        return cls(factory.from_url(url, **options), max_file_bytes=max_file_bytes)

    def get(self, key: str) -> bytes | None:
        value = self.client.get(key)
        if value is not None and (not isinstance(value, bytes) or len(value) > self.max_file_bytes):
            raise ValueError("Redis object exceeds binary read budget")
        return value

    def immutable(self, key: str, data: bytes) -> None:
        if len(data) > self.max_file_bytes:
            raise ValueError("Redis object exceeds write budget")
        try:
            self.client.set(key, data, nx=True)
        except Exception:
            if self.get(key) != data:
                raise
        if self.get(key) != data:
            raise ConflictError("immutable Redis content collision")

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def commit(
        self, key: str, expected: bytes | None, value: bytes, ready: dict[str, bytes]
    ) -> None:
        if len(value) > self.max_file_bytes:
            raise ValueError("Redis catalog exceeds budget")
        keys = [key, *ready]
        args = [expected or b"", value, *ready.values()]
        try:
            result = self.client.eval(COMMIT_CATALOG, len(keys), *keys, *args)
        except Exception:
            if self.get(key) == value and all(self.get(k) == v for k, v in ready.items()):
                return  # Lost acknowledgement, verified by exact operation bytes.
            raise
        if result != 1:
            raise ConflictError("Redis catalog or ready changed")
