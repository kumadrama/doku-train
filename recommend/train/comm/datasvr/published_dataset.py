"""Versioned reader for published offline manifests and immutable local caches."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from recommend.train.comm.runtime_budget import check_memory


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


class PublishedDataset:
    def __init__(self, config: dict[str, Any], *, local_root: Path | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.options = self.config["dataset"]
        self.local_root = local_root.resolve() if local_root else None
        self.cache = Path(config["execution"]["cache_dir"]).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client: Any = None
        if "daily_prefix" in self.options:
            if local_root is not None:
                raise ValueError("daily input does not support a range-manifest local mirror")
            self._freeze_daily()
        self.manifest_uri = self.options["manifest_uri"]
        self.source_prefix = self.manifest_uri.rsplit("/", 1)[0] + "/"
        body = self._read(self.manifest_uri)
        self.manifest = json.loads(body)
        m = self.manifest
        if m.get("manifest_version") != "doku-train-dataset-manifest/v1":
            raise ValueError("unsupported manifest version")
        logical = digest({k: v for k, v in m.items() if k != "content_sha256"})
        if logical != m["content_sha256"] or logical != self.options["manifest_sha256"]:
            raise ValueError("manifest digest mismatch")
        for source, key in [
            ("dataset_version", "version"),
            ("feature_schema_version", "feature_schema_version"),
            ("label_definition_version", "label_definition_version"),
        ]:
            if m[source] != self.options[key]:
                raise ValueError(f"dataset identity mismatch: {source}")
        parts = m["partitions"]
        if sum(p["rows"] for p in parts) != m["row_count"] or len({p["uri"] for p in parts}) != len(
            parts
        ):
            raise ValueError("invalid manifest partition counts")
        self.contract_digests = {}
        contracts = {}
        for name in ["features", "labels"]:
            raw = self._read(self.source_prefix + f"contracts/{name}.json")
            self.contract_digests[name] = hashlib.sha256(raw).hexdigest()
            if (
                m.get("contract_sha256", {}).get(name, self.contract_digests[name])
                != self.contract_digests[name]
            ):
                raise ValueError("frozen contract checksum mismatch")
            contracts[name] = json.loads(raw)
        if (
            contracts["features"]["feature_spec_version"] != m["feature_schema_version"]
            or m["label_definition_version"] not in contracts["labels"]["label_definition_versions"]
        ):
            raise ValueError("source contract versions differ")
        self.dtypes = {x["name"]: x["dtype"] for x in contracts["features"]["features"]}
        if not set(config["features"]) <= self.dtypes.keys():
            raise ValueError("feature is not in source contract")
        self.labels = set(contracts["labels"]["labels"])
        # The v1 contract has no maturity field; external audits are run evidence.
        # Dataset names must not act as an allowlist or imply validation.
        self.maturity_status = "not_declared_by_source_manifest"
        pa.set_cpu_count(config["execution"]["arrow_threads"])
        self._verified: dict[str, Path] = {}

    def _freeze_daily(self) -> None:
        from recommend.train.comm.datasvr.daily_input import resolve_days

        self.source_prefix = self.options["daily_prefix"].rstrip("/") + "/"
        dates: set[date] = set()
        for split in ["train", "validation", "test"]:
            start, end = [
                date.fromisoformat(self.config["split"][f"{split}_{k}"]) for k in ["from", "to"]
            ]
            dates.update(start + timedelta(days=i) for i in range((end - start).days + 1))
        manifest, contracts = resolve_days(self.options, sorted(dates), self._read)
        target = Path(self.config["output"]["root"]) / self.config["output"]["run_id"] / "input"
        target.mkdir(parents=True, exist_ok=False)
        (target / "contracts").mkdir()
        for name, body in contracts.items():
            with (target / "contracts" / f"{name}.json").open("xb") as f:
                f.write(body)
        with (target / "manifest.json").open("x") as f:
            json.dump(manifest, f, sort_keys=True)
        self.options.pop("daily_prefix")
        self.options.update(
            manifest_uri=(target / "manifest.json").resolve().as_uri(),
            manifest_sha256=manifest["content_sha256"],
            version=manifest["dataset_version"],
        )
        self.options["allowed_prefixes"].append(target.resolve().as_uri() + "/")

    def _local_path(self, uri: str) -> Path | None:
        parsed = urlsplit(uri)
        if parsed.query or parsed.fragment:
            raise ValueError("input URI query and fragment are forbidden")
        if not any(uri.startswith(p) for p in self.options["allowed_prefixes"]):
            raise ValueError("input outside configured allowed prefixes")
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
            roots = [
                Path(unquote(urlsplit(p).path)).resolve()
                for p in self.options["allowed_prefixes"]
                if urlsplit(p).scheme == "file"
            ]
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError("local input outside allowed prefix")
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

            self.client = boto3.Session(profile_name=self.options["aws_profile"]).client("s3")
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
        parts = [
            p
            for p in self.manifest["partitions"]
            if start <= date.fromisoformat(p["event_time_from"][:10]) <= end
        ]
        days = {date.fromisoformat(p["event_time_from"][:10]) for p in parts}
        if days != {start + timedelta(days=i) for i in range((end - start).days + 1)}:
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
                    self._read(self.manifest_uri)
                u = urlsplit(p["uri"])
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
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != p["sha256"]:
                raise ValueError("shard checksum mismatch")
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
                    pending.append((p, pool.submit(self._shard, p)))
            while pending:
                p, future = pending.popleft()
                path = future.result()
                next_part = next(source, None)
                if next_part is not None:
                    pending.append((next_part, pool.submit(self._shard, next_part)))
                yield from self._rows_in_shard(p, path, columns)

    def _rows_in_shard(
        self, p: dict[str, Any], path: Path, columns: set[str]
    ) -> Iterator[dict[str, Any]]:
        file = pq.ParquetFile(path)
        # Published v1 partitions are keyed by exposure time, not request time.
        time_column = "impression_time"
        selected = sorted(columns | {time_column})
        if not set(selected) <= set(file.schema_arrow.names):
            raise ValueError(
                f"missing required columns: {set(selected) - set(file.schema_arrow.names)}"
            )
        start, end = [
            datetime.fromisoformat(p[f"event_time_{x}"].replace("Z", "+00:00")).timestamp() * 1000
            for x in ["from", "to"]
        ]
        for batch in file.iter_batches(
            batch_size=self.config["batch_size"],
            columns=selected,
            use_threads=self.config["execution"]["arrow_threads"] > 1,
        ):
            check_memory(batch.nbytes * 3)
            for row in batch.to_pylist():
                value = row[time_column]
                if isinstance(value, str):
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                timestamp = value.timestamp() * 1000 if isinstance(value, datetime) else value
                if timestamp is None or not start <= timestamp < end:
                    raise ValueError("row outside declared event partition")
                yield row
