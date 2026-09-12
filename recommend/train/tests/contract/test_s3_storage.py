from __future__ import annotations

import hashlib
from io import BytesIO

import pytest

from recommend.train.comm.datasvr.s3_storage import S3Storage


class Body:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def read(self, amount: int) -> bytes:
        return self.value[:amount]


class FakeS3Client:
    def __init__(self) -> None:
        self.objects = {("allowed", "training/input.bin"): b"input"}
        self.calls: list[str] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.calls.append("head_object")
        value = self.objects[(Bucket, Key)]
        return {"ContentLength": len(value), "ChecksumSHA256": None}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.calls.append("get_object")
        return {"Body": Body(self.objects[(Bucket, Key)])}

    def download_fileobj(self, bucket: str, key: str, sink: BytesIO) -> None:
        self.calls.append("download_fileobj")
        sink.write(self.objects[(bucket, key)])

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str) -> None:
        self.calls.append("put_object")
        assert IfNoneMatch == "*"
        identity = (Bucket, Key)
        if identity in self.objects:
            raise FileExistsError(Key)
        self.objects[identity] = Body


def test_s3_storage_uses_exact_keys_without_listing() -> None:
    client = FakeS3Client()
    store = S3Storage(client, allowed_bucket="allowed", allowed_prefix="training/")
    assert store.get_bytes("s3://allowed/training/input.bin", max_bytes=5) == b"input"
    sink = BytesIO()
    store.download_to("s3://allowed/training/input.bin", sink)
    head = store.put_bytes_if_absent("s3://allowed/training/output.bin", b"output")
    assert head.sha256 == hashlib.sha256(b"output").hexdigest()
    assert "list_objects" not in " ".join(client.calls)


def test_s3_storage_rejects_bucket_or_prefix_escape() -> None:
    client = FakeS3Client()
    store = S3Storage(client, allowed_bucket="allowed", allowed_prefix="training/")
    with pytest.raises(ValueError, match="bucket"):
        store.head("s3://other/training/input.bin")
    with pytest.raises(ValueError, match="prefix"):
        store.head("s3://allowed/private/input.bin")
