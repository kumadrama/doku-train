"""Versioned reader for published offline manifests and immutable local caches."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from recommend.train.comm import timing
from recommend.train.comm.datasvr.daily_contracts import (
    digest as digest,
)
from recommend.train.comm.datasvr.daily_contracts import (
    validate_contracts,
    verify_daily_evidence,
)
from recommend.train.comm.runtime_budget import check_memory


class PublishedDataset:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        local_root: Path | None = None,
        frozen_input: Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.options = self.config["dataset"]
        self.local_root = local_root.resolve() if local_root else None
        self.timer = timing.current()
        self.cache = Path(config["execution"]["cache_dir"]).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client: Any = None
        self.manifest_uri = self.options["manifest_uri"]
        self.source_prefix = self.manifest_uri.rsplit("/", 1)[0] + "/"
        frozen = json.loads((frozen_input / "record.json").read_text()) if frozen_input else None
        if frozen is not None and frozen["manifest_uri"] != self.manifest_uri:
            raise ValueError("frozen input source differs")
        body = (
            (frozen_input / "manifest.json").read_bytes()
            if frozen_input
            else self._read(self.manifest_uri)
        )
        self.manifest_bytes = body
        if (
            frozen is not None
            and hashlib.sha256(body).hexdigest() != frozen["manifest_file_sha256"]
        ):
            raise ValueError("frozen manifest checksum mismatch")
        self.manifest = json.loads(body)
        m = self.manifest
        if m.get("manifest_version") != "doku-train-dataset-manifest/v1":
            raise ValueError("unsupported manifest version")
        logical = digest({k: v for k, v in m.items() if k != "content_sha256"})
        if logical != m["content_sha256"]:
            raise ValueError("manifest digest mismatch")
        for key in [
            "dataset_id",
            "dataset_version",
            "feature_schema_version",
            "label_definition_version",
        ]:
            if not isinstance(m.get(key), str) or not m[key]:
                raise ValueError(f"missing manifest identity: {key}")
        if not isinstance(m.get("as_of_ms"), int):
            raise ValueError("missing manifest cutoff")
        parts = m["partitions"]
        if sum(p["rows"] for p in parts) != m["row_count"] or len({p["uri"] for p in parts}) != len(
            parts
        ):
            raise ValueError("invalid manifest partition counts")
        for part in parts:
            self._local_path(part["uri"])
            if part["rows"] < 0 or len(part["sha256"]) != 64:
                raise ValueError("invalid partition metadata")
            start, end = [
                datetime.fromisoformat(part[f"event_time_{edge}"].replace("Z", "+00:00"))
                for edge in ["from", "to"]
            ]
            if start.tzinfo is None or end.tzinfo is None or start >= end:
                raise ValueError("invalid partition times")
        self.contract_digests = {}
        self.contract_bytes: dict[str, bytes] = {}
        contracts = {}
        for name in ["features", "labels"]:
            raw = (
                (frozen_input / "contracts" / f"{name}.json").read_bytes()
                if frozen_input
                else self._read(self.source_prefix + f"contracts/{name}.json")
            )
            self.contract_bytes[name] = raw
            self.contract_digests[name] = hashlib.sha256(raw).hexdigest()
            if (
                m.get("contract_sha256", {}).get(name, self.contract_digests[name])
                != self.contract_digests[name]
            ):
                raise ValueError("frozen contract checksum mismatch")
            if (
                frozen is not None
                and self.contract_digests[name] != frozen["contract_digests"][name]
            ):
                raise ValueError("frozen contract checksum mismatch")
            contracts[name] = json.loads(raw)
        validate_contracts(m, contracts)
        self.dtypes = {x["name"]: x["dtype"] for x in contracts["features"]["features"]}
        if not set(config["features"]) <= self.dtypes.keys():
            raise ValueError("feature is not in source contract")
        self.daily_fields: dict[str, set[str]] = {}
        self.source_records: list[dict[str, Any]] = []
        if "daily_contracts_version" in m:
            if m["daily_contracts_version"] != "doku-daily-contracts/v1":
                raise ValueError("unsupported daily contracts version")
            self.daily_fields, self.source_records = verify_daily_evidence(m, contracts)
            selected = set(config["features"])
            defaults = config["feature_default_values"]
            for fields in self.daily_fields.values():
                missing = selected - fields
                if missing - defaults.keys():
                    raise ValueError(
                        f"missing explicit feature defaults: {sorted(missing - defaults.keys())}"
                    )
            for record in self.source_records:
                fields = next(
                    self.daily_fields[p["uri"]]
                    for p in m["partitions"]
                    if p["event_time_from"][:10] == record["date"]
                )
                record["defaulted_features"] = sorted(selected - fields)
        self.labels = set(contracts["labels"]["labels"])
        # The v1 contract has no maturity field; external audits are run evidence.
        # Dataset names must not act as an allowlist or imply validation.
        self.maturity_status = "not_declared_by_source_manifest"
        pa.set_cpu_count(config["execution"]["arrow_threads"])
        self._verified: dict[str, Path] = {}

    def freeze(self, target: Path) -> None:
        """Persist exactly the metadata read for this run, without rereading its source."""
        target.mkdir(parents=True, exist_ok=False)
        (target / "contracts").mkdir()
        (target / "manifest.json").write_bytes(self.manifest_bytes)
        for name, raw in self.contract_bytes.items():
            (target / "contracts" / f"{name}.json").write_bytes(raw)
        record = {
            "manifest_uri": self.manifest_uri,
            "manifest_file_sha256": hashlib.sha256(self.manifest_bytes).hexdigest(),
            "manifest_sha256": self.manifest["content_sha256"],
            "contract_digests": self.contract_digests,
        }
        (target / "record.json").write_text(json.dumps(record, sort_keys=True))

    def _local_path(self, uri: str) -> Path | None:
        parsed = urlsplit(uri)
        if parsed.query or parsed.fragment:
            raise ValueError("input URI query and fragment are forbidden")
        if parsed.username or parsed.password or ".." in unquote(parsed.path).split("/"):
            raise ValueError("invalid input URI path")
        if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
            raise ValueError("invalid S3 URI")
        if urlsplit(self.manifest_uri).scheme == "s3" and parsed.scheme != "s3":
            raise ValueError("remote manifest cannot reference local files")
        if self.local_root is not None:
            if not uri.startswith(self.source_prefix):
                raise ValueError("local mirror must match manifest prefix")
            path = (self.local_root / uri[len(self.source_prefix) :]).resolve()
            if not path.is_relative_to(self.local_root):
                raise ValueError("invalid mirror path")
            return path
        u = urlsplit(uri)
        if u.scheme == "file":
            if u.netloc not in ("", "localhost"):
                raise ValueError("invalid local URI host")
            path = Path(unquote(u.path)).resolve()
            return path
        if u.scheme != "s3" or u.query or u.fragment:
            raise ValueError("unsupported input URI")
        return None

    def _read(self, uri: str) -> bytes:
        path = self._local_path(uri)
        limit = self.config["execution"]["memory_limit_bytes"]
        if path is not None:
            if path.stat().st_size > limit:
                raise ValueError("input exceeds memory budget")
            return path.read_bytes()
        if self.client is None:
            import boto3

            self.client = boto3.Session().client("s3")
        u = urlsplit(uri)
        response = self.client.get_object(Bucket=u.netloc, Key=u.path[1:])
        with response["Body"] as stream:
            body: bytes = stream.read(limit + 1)
        if len(body) > limit:
            raise ValueError("input exceeds memory budget")
        return body

    def partitions(self, split: str) -> list[dict[str, Any]]:
        opts = self.config["split"]
        start, end = [date.fromisoformat(opts[f"{split}_{x}"]) for x in ["from", "to"]]
        expected = (
            {date.fromisoformat(d) for d in opts[f"{split}_dates"]}
            if f"{split}_dates" in opts
            else {start + timedelta(days=i) for i in range((end - start).days + 1)}
        )
        parts = [
            p
            for p in self.manifest["partitions"]
            if date.fromisoformat(p["event_time_from"][:10]) in expected
        ]
        days = {date.fromisoformat(p["event_time_from"][:10]) for p in parts}
        if days != expected:
            raise ValueError("requested dates are not completely covered")
        return sorted(parts, key=lambda p: (p["event_time_from"], p["uri"]))

    def _shard(self, p: dict[str, Any]) -> Path:
        if p["uri"] in self._verified:
            return self._verified[p["uri"]]
        path = self._local_path(p["uri"])
        if path is None:
            path = self.cache / (p["sha256"] + ".parquet")
            if not path.exists():
                # Stream transfer with explicit budget and atomic cache publication.
                if self.client is None:
                    import boto3

                    self.client = boto3.Session().client("s3")
                u = urlsplit(p["uri"])
                with self.timer.span("shard_download", unit="shards"):
                    response = self.client.get_object(Bucket=u.netloc, Key=u.path[1:])
                    size = int(response["ContentLength"])
                    used = sum(x.stat().st_size for x in self.cache.glob("*") if x.is_file())
                    if used + size > self.config["execution"]["cache_limit_bytes"]:
                        response["Body"].close()
                        raise ValueError("cache disk budget exceeded")
                    temp = path.with_suffix(".partial")
                    try:
                        with response["Body"] as stream, temp.open("xb") as out:
                            while chunk := stream.read(1024 * 1024):
                                out.write(chunk)
                        temp.rename(path)
                    finally:
                        temp.unlink(missing_ok=True)
        with self.timer.span("shard_checksum", unit="shards"), path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != p["sha256"]:
                raise ValueError("shard checksum mismatch")
        with self.timer.span("shard_metadata", unit="shards"):
            if pq.ParquetFile(path).metadata.num_rows != p["rows"]:
                raise ValueError("shard row count mismatch")
        self._verified[p["uri"]] = path
        return path

    def rows(self, split: str, columns: set[str]) -> Iterator[dict[str, Any]]:
        workers = self.config["execution"]["reader_workers"]
        parts = self.partitions(split)
        if workers == 0:
            for part in parts:
                yield from self._rows_in_shard(part, self._shard(part), columns)
            return
        with ThreadPoolExecutor(
            max_workers=max(1, min(workers, self.config["execution"]["s3_transfer_threads"]))
        ) as pool:
            pending: deque[tuple[dict[str, Any], Future[Path]]] = deque()
            source = iter(parts)
            depth = self.config["execution"]["prefetch_batches"] if workers else 1
            for _ in range(depth):
                p = next(source, None)
                if p is not None:
                    pending.append((p, pool.submit(copy_context().run, self._shard, p)))
            while pending:
                p, future = pending.popleft()
                with self.timer.span("prefetch_wait"):
                    path = future.result()
                next_part = next(source, None)
                if next_part is not None:
                    pending.append(
                        (next_part, pool.submit(copy_context().run, self._shard, next_part))
                    )
                yield from self._rows_in_shard(p, path, columns)

    def _rows_in_shard(
        self, p: dict[str, Any], path: Path, columns: set[str]
    ) -> Iterator[dict[str, Any]]:
        file = pq.ParquetFile(path)
        # Published v1 partitions are keyed by exposure time, not request time.
        time_column = "impression_time"
        requested = columns | {time_column}
        defaults = self.config["feature_default_values"]
        absent = (
            (requested & set(self.config["features"])) - self.daily_fields[p["uri"]]
            if self.daily_fields
            else set()
        )
        selected = sorted(requested - absent)
        if not set(selected) <= set(file.schema_arrow.names):
            raise ValueError(
                f"missing required columns: {set(selected) - set(file.schema_arrow.names)}"
            )
        start, end = [
            datetime.fromisoformat(p[f"event_time_{x}"].replace("Z", "+00:00")).timestamp() * 1000
            for x in ["from", "to"]
        ]
        for batch in self.timer.iterate(
            "parquet_batch",
            file.iter_batches(
                batch_size=self.config["batch_size"],
                columns=selected,
                use_threads=self.config["execution"]["arrow_threads"] > 1,
            ),
        ):
            check_memory(batch.nbytes * 3)
            with self.timer.span("to_pylist", count=batch.num_rows, unit="rows"):
                rows = batch.to_pylist()
            for row in rows:
                with self.timer.span("row_validate", unit="rows"):
                    value = row[time_column]
                    if isinstance(value, str):
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    timestamp = value.timestamp() * 1000 if isinstance(value, datetime) else value
                    if timestamp is None or not start <= timestamp < end:
                        raise ValueError("row outside declared event partition")
                    row.update({name: copy.deepcopy(defaults[name]) for name in absent})
                yield row
