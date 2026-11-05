"""Run-scoped basic input preparation, with no final feature/label encoding."""

import ast
import hashlib
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.common_utils import eval_expr

from .. import timing
from ..runtime_budget import check_memory
from .published_dataset import stream_digest


def file_checksum(path):
    with path.open("rb") as stream:
        return stream_digest(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def organize(row, defaults, groups, histories):
    row = dict(row)
    for group in groups.values():
        if "weight_feature" in group:
            ids, weights = row.get(group["features"][0]), row.get(group["weight_feature"])
            if (ids is None) != (weights is None) or (ids is not None and len(ids) != len(weights)):
                raise ValueError("Unaligned history ID/weight")
    for key, value in defaults.items():
        if row.get(key) is None:
            row[key] = value.copy() if isinstance(value, list) else value
    for name, group in groups.items():
        if not group.get("history_pooling"):
            continue
        fields = group["features"]
        ids = row[fields[0]]
        if ids is None or any(x is None for x in ids):
            raise ValueError("History must be defaulted and contain no null elements")
        if any(len(row[f]) != len(ids) for f in fields):
            raise ValueError("Unaligned history fields")
        opt = histories.get(name)
        if opt:
            cap = opt["max_hist_size"]
            if cap <= 0:
                raise ValueError("History cap must be positive")
            if opt["truncate"] == "tail" and opt["duplicates"] == "preserve":
                for f in fields + ([group["weight_feature"]] if "weight_feature" in group else []):
                    row[f] = row[f][-cap:]
            elif (
                opt["truncate"] == "top_weight"
                and opt["duplicates"] == "sum_weights"
                and opt["tie_break"] == "raw_id_ascending"
            ):
                if len(fields) != 1:
                    raise ValueError("Weighted history requires one ID field")
                merged = {}
                for key, weight in zip(ids, row[group["weight_feature"]]):
                    if not math.isfinite(weight) or weight < 0:
                        raise ValueError("History weight must be finite and nonnegative")
                    merged[key] = merged.get(key, 0.0) + weight
                pairs = sorted(merged.items(), key=lambda p: (-p[1], p[0]))[:cap]
                if any(not math.isfinite(v) for _, v in pairs):
                    raise ValueError("Merged history weight overflow")
                row[fields[0]] = [k for k, _ in pairs]
                row[group["weight_feature"]] = [v for _, v in pairs]
            else:
                raise ValueError("Unsupported history policy")
        row["length:" + name] = len(row[fields[0]])
    return row


def organize_columns(columns, defaults, groups, histories):
    """Keep ordinary fields columnar; reuse the exact row history rules."""
    size = len(next(iter(columns.values())))
    if any(len(values) != size for values in columns.values()):
        raise ValueError("Unaligned input columns")
    history_fields = {
        field
        for group in groups.values()
        if group.get("history_pooling") or "weight_feature" in group
        for field in group["features"]
        + ([group["weight_feature"]] if "weight_feature" in group else [])
    }
    result = dict(columns)
    for key, default in defaults.items():
        if key not in history_fields:
            result[key] = [
                (default.copy() if isinstance(default, list) else default)
                if value is None
                else value
                for value in columns.get(key, [None] * size)
            ]
    history_defaults = {k: v for k, v in defaults.items() if k in history_fields}
    outputs = history_fields | {
        "length:" + name for name, group in groups.items() if group.get("history_pooling")
    }
    collected = {key: [] for key in outputs}
    for index in range(size):
        row = organize(
            {k: columns[k][index] for k in history_fields if k in columns},
            history_defaults,
            groups,
            histories,
        )
        for key in outputs:
            collected[key].append(row[key])
    result.update(collected)
    return result


@timing.measured("prepare")
def prepare(
    blocks,
    root,
    defaults,
    params,
    histories,
    item_filter,
    identity,
    budget,
    *,
    allow_empty=False,
    budget_root=None,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    files = []
    total = 0
    used = sum(p.stat().st_size for p in Path(budget_root or root.parent).rglob("*.parquet"))
    for block in blocks:
        check_memory()
        if isinstance(block, Mapping):
            if not block or not len(next(iter(block.values()))):
                continue
            columns = organize_columns(block, defaults, params.input_features, histories)
        else:
            organized = [organize(row, defaults, params.input_features, histories) for row in block]
            if not organized:
                continue
            columns = {k: [row[k] for row in organized] for k in organized[0]}
        # Keep Finder expression evaluation and validation-before-filter order.
        with tf.Graph().as_default(), tf.Session() as sess:
            names = {n.id for n in ast.walk(ast.parse(item_filter)) if isinstance(n, ast.Name)}
            values = {k: tf.constant(columns[k]) for k in names if k in columns}
            keep = sess.run(tf.convert_to_tensor(eval_expr(item_filter, values)))
        keep = np.broadcast_to(keep, (len(next(iter(columns.values()))),))
        indices = np.flatnonzero(keep)
        if not len(indices):
            continue
        table = pa.Table.from_pydict({k: [v[i] for i in indices] for k, v in columns.items()})
        path = root / f"{len(files):06d}.parquet"
        pq.write_table(table, path)
        used += path.stat().st_size
        if used > budget:
            raise MemoryError("Prepared input disk budget exceeded")
        files.append(
            {
                "file": path.name,
                "rows": table.num_rows,
                "sha256": file_checksum(path),
            }
        )
        total += table.num_rows
    if not total and not allow_empty:
        raise ValueError("No samples after filter")
    manifest = {"identity": identity, "files": files, "rows": total}
    (root / "complete.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


def input_columns(params, dtypes):
    """Only graph inputs and history lengths are needed after time splitting."""
    return sorted(
        set(dtypes)
        | {
            "length:" + name
            for name, group in params.input_features.items()
            if group.get("history_pooling")
        }
    )


class ColumnRow(Mapping):
    """A row reference keeps its immutable NumPy columns alive through shuffle."""

    __slots__ = ("columns", "index", "block_key")

    def __init__(self, columns, index, block_key=None):
        self.columns = columns
        self.index = index
        self.block_key = block_key

    def __getitem__(self, key):
        return self.columns[key][self.index]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.columns)


def column_values(array):
    # Preserve Python None and nested null semantics outside the prepared fast path.
    if array.null_count or (pa.types.is_list(array.type) and array.values.null_count):
        return array.to_pylist()
    if pa.types.is_list(array.type):
        value_type = array.type.value_type
        if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
            return array.to_pylist()
    elif not (
        pa.types.is_integer(array.type)
        or pa.types.is_floating(array.type)
        or pa.types.is_boolean(array.type)
        or pa.types.is_string(array.type)
    ):
        return array.to_pylist()
    return array.to_numpy(zero_copy_only=False)


def rows(root, identity, columns=None, seed=None):
    root = Path(root)
    manifest = json.loads((root / "complete.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("Prepared identity mismatch")

    def read(records):
        for record in records:
            path = (root / record["file"]).resolve()
            boundary = root.parent.resolve() if "source_id" in record else root.resolve()
            if not path.is_relative_to(boundary) or file_checksum(path) != record["sha256"]:
                raise ValueError("Prepared shard mismatch")
            actual = 0
            for block_index, block in enumerate(
                pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns)
            ):
                values = {
                    name: column_values(array)
                    for name, array in zip(block.schema.names, block.columns)
                }
                for index in range(block.num_rows):
                    actual += 1
                    yield ColumnRow(
                        values, index, (identity, str(path), record["sha256"], block_index)
                    )
            if actual != record["rows"]:
                raise ValueError("Prepared shard row mismatch")

    files = manifest["files"]
    count = 0
    if seed is None or not files or not all("source_id" in r for r in files):
        for row in read(files):
            count += 1
            yield row
    else:
        groups = {}
        for record in files:
            groups.setdefault(record["source_id"], []).append(record)
        streams = [iter(read(records)) for records in groups.values()]
        remaining = [sum(r["rows"] for r in records) for records in groups.values()]
        rng = random.Random(seed)
        while sum(remaining):
            draw = rng.randrange(sum(remaining))
            for index, available in enumerate(remaining):
                if draw < available:
                    try:
                        row = next(streams[index])
                    except StopIteration as exc:
                        raise ValueError("Prepared source row mismatch") from exc
                    remaining[index] -= 1
                    count += 1
                    yield row
                    break
                draw -= available
        # Exhaust generators so final shard counts/checksums are checked too.
        for stream in streams:
            if next(stream, None) is not None:
                raise ValueError("Prepared source row mismatch")
    if count != manifest["rows"]:
        raise ValueError("Prepared total row mismatch")


def batches(source, size, seed=None, buffer_size=1, replacement=False):
    if size < 1 or buffer_size < 1:
        raise ValueError("Batch and shuffle capacity must be positive")
    rng = random.Random(seed)
    buffer = []
    batch = []
    for index_in_source, row in enumerate(source):
        if index_in_source % 1024 == 0:
            check_memory()
        if not replacement:
            buffer.append(row)
            if len(buffer) < buffer_size:
                continue
            index = rng.randrange(len(buffer)) if seed is not None else 0
            batch.append(buffer.pop(index))
        elif seed is None:
            batch.append(row)
        elif len(buffer) < buffer_size:
            buffer.append(row)
            continue
        else:
            index = rng.randrange(len(buffer))
            batch.append(buffer[index])
            buffer[index] = row
        if len(batch) == size:
            check_memory()
            yield batch
            batch = []
    while buffer:
        index = rng.randrange(len(buffer)) if seed is not None else 0
        if replacement:
            batch.append(buffer[index])
            buffer[index] = buffer[-1]
            buffer.pop()
        else:
            batch.append(buffer.pop(index))
        if len(batch) == size:
            check_memory()
            yield batch
            batch = []
    if batch:
        check_memory()
        yield batch


@timing.measured("tensor_batch")
def tensor_batch(batch, params, dtypes):
    item = {}
    history_fields = {
        f
        for g in params.input_features.values()
        if g.get("history_pooling")
        for f in g["features"] + ([g["weight_feature"]] if "weight_feature" in g else [])
    }
    for key, dtype in dtypes.items():
        if key not in history_fields:
            item[key] = np.asarray([row[key] for row in batch], dtype=dtype)
    result = {"item": item}
    for name, group in params.input_features.items():
        if not group.get("history_pooling"):
            continue
        source = group["from"]
        if source == "item":
            raise ValueError("History requires a separate feature group")
        lengths = np.asarray([row["length:" + name] for row in batch], np.int64)
        cap = max(1, int(lengths.max()))
        target = result.setdefault(source, {"length": lengths})
        for field in group["features"] + (
            [group["weight_feature"]] if "weight_feature" in group else []
        ):
            values = np.zeros((len(batch), cap), dtype=dtypes[field])
            for i, row in enumerate(batch):
                values[i, : len(row[field])] = row[field]
            target[field] = values
    return result


def column_batches(root, identity, columns, size):
    """Unshuffled fitting scan with the same cross-shard batch boundaries."""
    root = Path(root)
    manifest = json.loads((root / "complete.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("Prepared identity mismatch")
    pending = []
    buffered = total = 0
    for record in manifest["files"]:
        path = (root / record["file"]).resolve()
        boundary = root.parent.resolve() if "source_id" in record else root.resolve()
        if not path.is_relative_to(boundary) or file_checksum(path) != record["sha256"]:
            raise ValueError("Prepared shard mismatch")
        actual = 0
        for block in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns):
            check_memory()
            actual += block.num_rows
            total += block.num_rows
            offset = 0
            while offset < block.num_rows:
                take = min(size - buffered, block.num_rows - offset)
                pending.append(block.slice(offset, take))
                buffered += take
                offset += take
                if buffered == size:
                    yield join_columns(pending)
                    pending = []
                    buffered = 0
        if actual != record["rows"]:
            raise ValueError("Prepared shard row mismatch")
    if total != manifest["rows"]:
        raise ValueError("Prepared total row mismatch")
    if pending:
        yield join_columns(pending)


def join_columns(blocks):
    # Empty-history shards can have list<null> while later shards have list<int>.
    # Do not promote mixed numeric columns through Arrow (potential precision loss).
    result = {}
    for name in blocks[0].schema.names:
        parts = [column_values(block.column(name)) for block in blocks]
        if isinstance(parts[0], np.ndarray) and all(
            isinstance(part, np.ndarray) and part.dtype == parts[0].dtype for part in parts
        ):
            result[name] = np.concatenate(parts)
        else:
            result[name] = [value for part in parts for value in part]
    return result


@timing.measured("tensor_columns")
def tensor_columns(columns, params, dtypes):
    """Construct the same padded tensors without scalar row lookups."""
    history_fields = {
        field
        for group in params.input_features.values()
        if group.get("history_pooling")
        for field in group["features"]
        + ([group["weight_feature"]] if "weight_feature" in group else [])
    }
    result = {
        "item": {
            key: np.asarray(columns[key], dtype=dtype)
            for key, dtype in dtypes.items()
            if key not in history_fields
        }
    }
    for name, group in params.input_features.items():
        if not group.get("history_pooling"):
            continue
        source = group["from"]
        if source == "item":
            raise ValueError("History requires a separate feature group")
        lengths = np.asarray(columns["length:" + name], np.int64)
        cap = max(1, int(lengths.max()))
        target = result.setdefault(source, {"length": lengths})
        for field in group["features"] + (
            [group["weight_feature"]] if "weight_feature" in group else []
        ):
            values = np.zeros((len(lengths), cap), dtype=dtypes[field])
            for index, value in enumerate(columns[field]):
                values[index, : len(value)] = value
            target[field] = values
    return result
