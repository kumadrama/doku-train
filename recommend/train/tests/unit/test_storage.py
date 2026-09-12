from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.datasvr.storage import parse_uri


def test_parse_uri_rejects_queries_latest_and_unsupported_schemes() -> None:
    with pytest.raises(ValueError, match="query"):
        parse_uri("s3://bucket/key?token=secret")
    with pytest.raises(ValueError, match="latest"):
        parse_uri("s3://bucket/training/latest/manifest.json")
    with pytest.raises(ValueError, match="unsupported"):
        parse_uri("https://example.com/object")


def test_local_storage_reads_and_creates_without_overwrite(tmp_path) -> None:
    store = LocalStorage(tmp_path)
    uri = (tmp_path / "outputs" / "value.json").as_uri()
    head = store.put_bytes_if_absent(uri, b"payload")
    assert head.size_bytes == 7
    assert head.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert store.get_bytes(uri, max_bytes=7) == b"payload"
    sink = BytesIO()
    assert store.download_to(uri, sink) == head
    assert sink.getvalue() == b"payload"
    with pytest.raises(FileExistsError):
        store.put_bytes_if_absent(uri, b"replacement")


def test_local_storage_enforces_root_and_read_bound(tmp_path) -> None:
    store = LocalStorage(tmp_path / "allowed")
    outside = (tmp_path / "outside.txt").as_uri()
    with pytest.raises(ValueError, match="outside allowed root"):
        store.put_bytes_if_absent(outside, b"secret")
    inside = (tmp_path / "allowed" / "large.bin").as_uri()
    store.put_bytes_if_absent(inside, b"1234")
    with pytest.raises(ValueError, match="max_bytes"):
        store.get_bytes(inside, max_bytes=3)


def test_local_storage_head_hashes_without_reading_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"stream-me")
    store = LocalStorage(tmp_path)

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("head must not use Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    head = store.head(path.as_uri())
    assert head.size_bytes == len(b"stream-me")
    assert head.sha256 == hashlib.sha256(b"stream-me").hexdigest()
