from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from recommend.train.comm.datasvr.storage import parse_uri
from recommend.train.comm.model_params import StrictModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_manifest_sha256(payload: Mapping[str, object]) -> str:
    logical_payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_json(logical_payload).encode()).hexdigest()


class ObjectReference(StrictModel):
    uri: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_uri(self) -> ObjectReference:
        parse_uri(self.uri)
        return self


class ShardManifest(ObjectReference):
    row_count: int = Field(ge=0)
    event_date: date
    min_event_time_ms: int = Field(ge=0)
    max_event_time_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> ShardManifest:
        if self.min_event_time_ms > self.max_event_time_ms:
            raise ValueError("shard event time range is invalid")
        return self


class DatasetManifest(StrictModel):
    manifest_version: Literal["1"]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    feature_schema: ObjectReference
    label_definition_version: str = Field(min_length=1)
    label_definition: ObjectReference
    as_of_ms: int = Field(ge=0)
    label_maturity_hours: int = Field(ge=0)
    shards: tuple[ShardManifest, ...] = Field(min_length=1)
    row_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> DatasetManifest:
        if sum(shard.row_count for shard in self.shards) != self.row_count:
            raise ValueError("manifest row_count does not equal shard row_count sum")
        uris = [shard.uri for shard in self.shards]
        if len(set(uris)) != len(uris):
            raise ValueError("duplicate shard URI")
        expected = calculate_manifest_sha256(self.model_dump(mode="json"))
        if self.content_sha256 != expected:
            raise ValueError("manifest content_sha256 mismatch")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json_bytes(cls, body: bytes) -> DatasetManifest:
        return cls.model_validate_json(body)
