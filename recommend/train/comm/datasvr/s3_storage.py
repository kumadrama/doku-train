from __future__ import annotations

import base64
import hashlib
from typing import Any, BinaryIO

from recommend.train.comm.datasvr.storage import ObjectHead, ParsedUri, parse_uri


class _HashingSink:
    def __init__(self, sink: BinaryIO) -> None:
        self._sink = sink
        self._hash = hashlib.sha256()
        self.size_bytes = 0

    def write(self, body: bytes) -> int:
        written = self._sink.write(body)
        accepted = body[:written]
        self._hash.update(accepted)
        self.size_bytes += written
        return written

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


class S3Storage:
    def __init__(self, client: Any, *, allowed_bucket: str, allowed_prefix: str) -> None:
        self._client = client
        self._allowed_bucket = allowed_bucket
        self._allowed_prefix = allowed_prefix.strip("/") + "/"

    def _location(self, uri: str) -> ParsedUri:
        parsed = parse_uri(uri)
        if parsed.scheme != "s3":
            raise ValueError("S3Storage requires an s3 URI")
        if parsed.bucket != self._allowed_bucket:
            raise ValueError("s3 bucket is outside allowed bucket")
        if not parsed.key.startswith(self._allowed_prefix):
            raise ValueError("s3 key is outside allowed prefix")
        return parsed

    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        location = self._location(uri)
        response = self._client.get_object(Bucket=location.bucket, Key=location.key)
        body: bytes = response["Body"].read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("object exceeds max_bytes")
        return body

    def head(self, uri: str) -> ObjectHead:
        location = self._location(uri)
        response = self._client.head_object(Bucket=location.bucket, Key=location.key)
        encoded_checksum = response.get("ChecksumSHA256")
        checksum = None
        if encoded_checksum:
            checksum = base64.b64decode(encoded_checksum).hex()
        return ObjectHead(int(response["ContentLength"]), checksum)

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead:
        location = self._location(uri)
        hashing_sink = _HashingSink(sink)
        self._client.download_fileobj(location.bucket, location.key, hashing_sink)
        return ObjectHead(hashing_sink.size_bytes, hashing_sink.sha256)

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead:
        location = self._location(uri)
        self._client.put_object(
            Bucket=location.bucket,
            Key=location.key,
            Body=body,
            IfNoneMatch="*",
        )
        return ObjectHead(len(body), hashlib.sha256(body).hexdigest())
