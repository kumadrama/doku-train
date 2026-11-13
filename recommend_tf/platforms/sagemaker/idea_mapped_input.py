"""Freeze published Idea UTC days and project the two map-key history features."""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recommend_tf.train.comm.artifact_integrity import checksum
from recommend_tf.train.comm.datasvr.daily_contracts import digest

from .publish_model import immutable_put, location

MAPS = {
    "drama_interests_v1": "drama_interest_ids",
    "not_interested_dramas": "not_interested_drama_ids",
}


def dates(start, cutoff):
    end = date.fromisoformat(cutoff)
    first = max(date.fromisoformat(start), end - timedelta(days=30))
    if first >= end:
        raise ValueError("Idea requires a training date before validation")
    return [(first + timedelta(days=i)).isoformat() for i in range((end - first).days + 1)]


def project(src, dest):
    reader = pq.ParquetFile(src)
    schema = reader.schema_arrow
    for field, name in MAPS.items():
        kind = schema.field(field).type
        if not pa.types.is_map(kind) or not pa.types.is_int64(kind.key_type):
            raise ValueError("Idea history field must be an int64 keyed map")
        schema = schema.append(pa.field(name, pa.list_(pa.int64())))
    with pq.ParquetWriter(dest, schema, compression="zstd") as writer:
        for batch in reader.iter_batches(batch_size=2048):
            table = pa.Table.from_batches([batch])
            for field, name in MAPS.items():
                col = batch.column(field)
                table = table.append_column(
                    name, pa.ListArray.from_arrays(col.offsets, col.keys, mask=col.is_null())
                )
            writer.write_table(table)
    return reader.metadata.num_rows


def freeze(s3, cfg, days, root):
    source, target, kms = (cfg[k].rstrip("/") for k in ("source_uri", "input_uri", "kms_key"))
    folder = Path(root)
    parts, refs, contracts = [], [], {}
    max_asof = 0

    def read(uri):
        bucket, key = location(uri)
        with s3.get_object(Bucket=bucket, Key=key)["Body"] as stream:
            return stream.read()

    def put(path, uri):
        immutable_put(s3, uri, path, kms)

    for day in days:
        prefix = source + "/date=" + day + "/"
        pointer = json.loads(read(prefix + "_SUCCESS"))
        uri = pointer["manifest_uri"]
        if not uri.startswith(prefix + "run_id="):
            raise ValueError("Idea marker points outside its day")
        body = read(uri)
        if hashlib.sha256(body).hexdigest() != pointer["manifest_sha256"]:
            raise ValueError("Idea manifest digest mismatch")
        m = json.loads(body)
        if (
            m["dataset_id"] != "idea-rank-mapped/v1"
            or digest({k: v for k, v in m.items() if k != "content_sha256"}) != m["content_sha256"]
        ):
            raise ValueError("unexpected mapped manifest")
        base = uri.rsplit("/", 1)[0] + "/"
        for name in ("features", "labels"):
            raw = read(base + "contracts/" + name + ".json")
            if (
                hashlib.sha256(raw).hexdigest() != m["contract_sha256"][name]
                or contracts.setdefault(name, raw) != raw
            ):
                raise ValueError("mapped contract differs from frozen window")
        refs.append(dict(date=day, **pointer))
        day_rows = 0
        for part in m["partitions"]:
            if not part["uri"].startswith(base) or part["event_time_from"][:10] != day:
                raise ValueError("unexpected mapped partition")
            src, dest = folder / "source.parquet", folder / "projected.parquet"
            bucket, key = location(part["uri"])
            s3.download_file(bucket, key, str(src))
            if checksum(src) != part["sha256"]:
                raise ValueError("mapped shard checksum mismatch")
            rows = project(src, dest)
            if rows != part["rows"]:
                raise ValueError("projected row count differs")
            sha = checksum(dest)
            dest_uri = target + "/shards/" + sha + ".parquet"
            put(dest, dest_uri)
            day_rows += rows
            parts.append(dict(part, uri=dest_uri, sha256=sha))
        if day_rows != m["row_count"] or not day_rows:
            raise ValueError("invalid mapped daily row count")
        max_asof = max(max_asof, m["as_of_ms"])
    features = json.loads(contracts["features"])
    features["features"].extend(
        dict(name=name, source=f"keys({field})", dtype="list<int64>")
        for field, name in MAPS.items()
    )
    features["feature_spec_version"] = "idea-mapped-drama-keys/" + digest(features["features"])
    contracts["features"] = json.dumps(features, sort_keys=True).encode()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in contracts.items()}
    merged = {k: m[k] for k in ("manifest_version", "dataset_id", "label_definition_version")}
    merged.update(
        dataset_version="mapped-keys-" + digest(parts),
        as_of_ms=max_asof,
        feature_schema_version=features["feature_spec_version"],
        partitions=parts,
        original_sources=refs,
        contract_sha256=hashes,
        row_count=sum(p["rows"] for p in parts),
    )
    merged["content_sha256"] = digest(merged)
    prefix = target + "/windows/" + merged["content_sha256"] + "/"
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(merged, sort_keys=True))
    put(manifest, prefix + "manifest.json")
    return merged, contracts, prefix + "manifest.json"
