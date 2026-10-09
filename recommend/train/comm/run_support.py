from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, ObjectReference
from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.datasvr.s3_storage import S3Storage
from recommend.train.comm.datasvr.storage import ObjectStore, parse_uri

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CONTRACT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    manifest: DatasetManifest
    store: ObjectStore


def build_input_store(uri: str) -> ObjectStore:
    parsed = parse_uri(uri)
    if parsed.scheme == "file":
        return LocalStorage(Path(parsed.key).parent)
    prefix = PurePosixPath(parsed.key).parent.as_posix()
    if prefix == ".":
        raise ValueError("s3 manifest must be under an explicit allowed prefix")
    import boto3

    return S3Storage(
        boto3.client("s3"),
        allowed_bucket=parsed.bucket or "",
        allowed_prefix=prefix,
    )


def build_output_store(uri: str) -> ObjectStore:
    parsed = parse_uri(uri)
    if parsed.scheme == "file":
        return LocalStorage(Path(parsed.key))
    import boto3

    return S3Storage(
        boto3.client("s3"),
        allowed_bucket=parsed.bucket or "",
        allowed_prefix=parsed.key,
    )


def _verify_reference(store: ObjectStore, reference: ObjectReference) -> None:
    if reference.size_bytes > MAX_CONTRACT_BYTES:
        raise ValueError("INPUT_CONTRACT_INVALID: contract object exceeds size limit")
    body = store.get_bytes(reference.uri, max_bytes=MAX_CONTRACT_BYTES)
    if len(body) != reference.size_bytes or hashlib.sha256(body).hexdigest() != reference.sha256:
        raise ValueError("DATA_INTEGRITY_MISMATCH: contract object")


def validate_dataset(manifest_uri: str) -> ValidatedDataset:
    store = build_input_store(manifest_uri)
    manifest = DatasetManifest.from_json_bytes(
        store.get_bytes(manifest_uri, max_bytes=MAX_MANIFEST_BYTES)
    )
    _verify_reference(store, manifest.feature_schema)
    _verify_reference(store, manifest.label_definition)
    for shard in manifest.shards:
        head = store.head(shard.uri)
        if head.size_bytes != shard.size_bytes or (
            head.sha256 is not None and head.sha256 != shard.sha256
        ):
            raise ValueError("DATA_INTEGRITY_MISMATCH: shard metadata")
    return ValidatedDataset(manifest=manifest, store=store)


def digest_json(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(body).hexdigest()


def source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def lock_digest() -> str:
    path = Path("uv.lock")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "unknown"
