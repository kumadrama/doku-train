"""Bounded S3 documents and immutable configuration snapshots."""

import json

from ..sagemaker.publish_model import location, missing
from .contracts import MAX_BYTES, canonical, digest


class Documents:
    def __init__(self, s3, group):
        self.s3 = s3
        self.prefix = group["storage_prefix"].rstrip("/")
        self.kms_key = group["kms_key"]

    def uri(self, key):
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError("invalid document key")
        return self.prefix + "/" + key

    def read(self, uri):
        bucket, key = location(uri)
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
        except Exception as error:
            if missing(error):
                return None, None
            raise
        with response["Body"] as stream:
            body = stream.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("oversized management document")
        return json.loads(body), response["ETag"]

    def put(self, uri, value, etag=None):
        bucket, key = location(uri)
        body = canonical(value)
        if len(body) > MAX_BYTES:
            raise ValueError("oversized management document")
        result = self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=self.kms_key,
            **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
        )
        return result["ETag"]

    def immutable(self, kind, value):
        sha = digest(value)
        uri = self.uri(kind + "/" + sha + ".json")
        try:
            self.put(uri, value)
        except Exception as error:
            if not conflict(error):
                raise
            if self.read(uri)[0] != value:
                raise ValueError("immutable management document differs") from error
        return {"uri": uri, "sha256": sha}

    def pinned(self, pin):
        value, _ = self.read(pin["uri"])
        if value is None or digest(value) != pin["sha256"]:
            raise ValueError("pinned management document differs")
        return value

    def list(self, key):
        yield from self.list_external(self.uri(key))

    def list_external(self, uri):
        bucket, prefix = location(uri)
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield "s3://" + bucket + "/" + item["Key"]


def conflict(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code") in {
        "PreconditionFailed",
        "ConditionalRequestConflict",
        "412",
        "409",
    }


class Journal:
    def __init__(self, documents, uri):
        self.documents, self.uri = documents, uri
        self.value, self.etag = documents.read(uri)

    def save(self):
        self.etag = self.documents.put(self.uri, self.value, self.etag)
