from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True, slots=True)
class ParsedUri:
    scheme: str
    bucket: str | None
    key: str


@dataclass(frozen=True, slots=True)
class ObjectHead:
    size_bytes: int
    sha256: str | None


class ObjectStore(Protocol):
    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes: ...

    def head(self, uri: str) -> ObjectHead: ...

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead: ...

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead: ...


def parse_uri(uri: str) -> ParsedUri:
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment:
        raise ValueError("object URI query and fragment are forbidden")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if "latest" in segments:
        raise ValueError("latest discovery is forbidden")
    if parsed.scheme == "s3":
        if not parsed.netloc or not segments:
            raise ValueError("s3 URI requires bucket and exact key")
        return ParsedUri("s3", parsed.netloc, "/".join(segments))
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost") or not parsed.path.startswith("/"):
            raise ValueError("file URI must be an absolute local path")
        return ParsedUri("file", None, unquote(parsed.path))
    raise ValueError(f"unsupported object URI scheme: {parsed.scheme or '<empty>'}")
