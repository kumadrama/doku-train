"""Optional ordered batch encoding, bounded per-run caching and CPU input prefetch."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import copy_context
from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np

from . import timing
from .datasvr.prepared import ColumnRow, tensor_batch, tensor_columns
from .runtime_budget import check_memory


def options(config):
    value = config.get("input_pipeline", {})
    allowed = {"batch_encoding", "encoded_cache_bytes", "prefetch_batches"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown input_pipeline options")
    result = {"batch_encoding": "rows", "encoded_cache_bytes": 0, "prefetch_batches": 0, **value}
    if result["batch_encoding"] not in ("rows", "columns"):
        raise ValueError("batch_encoding must be rows or columns")
    for name in ("encoded_cache_bytes", "prefetch_batches"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if result["encoded_cache_bytes"] and result["batch_encoding"] != "columns":
        raise ValueError("Encoded cache requires columns batch encoding")
    return result


def group_rows(batch):
    groups = {}
    for position, row in enumerate(batch):
        entry = groups.setdefault(id(row.columns), (row, [], []))
        entry[1].append(position)
        entry[2].append(row.index)
    return [
        (row, np.asarray(positions), np.asarray(indices))
        for row, positions, indices in groups.values()
    ]


def gather(groups, field, restore=None):
    pieces = []
    for row, target, indices in groups:
        values = row.columns[field]
        if isinstance(values, np.ndarray):
            pieces.append(values[indices])
        else:
            # Preserve ragged/null types instead of allowing NumPy numeric promotion.
            selected = np.empty(len(indices), dtype=object)
            for i, index in enumerate(indices):
                selected[i] = values[index]
            pieces.append(selected)
    if len({piece.dtype for piece in pieces}) > 1:
        # Match np.asarray(original Python values, requested dtype), without an
        # intermediate float promotion that would round large integer identifiers.
        pieces = [piece.astype(object) for piece in pieces]
    if restore is None:
        restore = np.argsort(np.concatenate([positions for _, positions, _ in groups]))
    return np.concatenate(pieces)[restore]


class InputPipeline:
    def __init__(self, config, params, dtypes):
        self.options = options(config)
        self.params = params
        self.dtypes = dtypes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.hits = self.misses = 0

    @timing.measured("encode_batch")
    def encode(self, batch):
        if self.options["batch_encoding"] == "rows" or not all(
            isinstance(row, ColumnRow) for row in batch
        ):
            return tensor_batch(batch, self.params, self.dtypes)
        groups = group_rows(batch)
        if not self.options["encoded_cache_bytes"]:
            restore = np.argsort(np.concatenate([positions for _, positions, _ in groups]))
            columns = {name: gather(groups, name, restore) for name in batch[0].columns}
            for name, dtype in self.dtypes.items():
                if dtype == "str":
                    # Match row encoding's batch-local Unicode width.
                    columns[name] = columns[name].tolist()
            return tensor_columns(columns, self.params, self.dtypes)
        return self.cached_batch(groups, len(batch))

    def encoded_block(self, row):
        key = row.block_key
        # References without immutable prepared provenance are never retained across calls.
        if key is not None and key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        value = tensor_columns(row.columns, self.params, self.dtypes)
        size = sum(array.nbytes for fields in value.values() for array in fields.values())
        if key is not None and size <= self.options["encoded_cache_bytes"]:
            while self.cache and self.cache_bytes + size > self.options["encoded_cache_bytes"]:
                _, (_, removed) = self.cache.popitem(last=False)
                self.cache_bytes -= removed
            check_memory()
            self.cache[key] = (value, size)
            self.cache_bytes += size
        return value

    def cached_batch(self, groups, count):
        blocks = [
            (positions, indices, self.encoded_block(row)) for row, positions, indices in groups
        ]
        positions = np.concatenate([p for p, _, _ in blocks])
        restore = np.argsort(positions)
        result = {"item": {}}
        for field in blocks[0][2]["item"]:
            values = np.concatenate(
                [block["item"][field][indices] for _, indices, block in blocks]
            )[restore]
            if self.dtypes[field] == "str":
                values = np.asarray(values.tolist(), dtype="str")
            result["item"][field] = values
        for name, group in self.params.input_features.items():
            if not group.get("history_pooling"):
                continue
            source = group["from"]
            lengths = np.asarray(gather(groups, "length:" + name), dtype=np.int64)
            cap = max(1, int(lengths.max()))
            target = result.setdefault(source, {"length": lengths})
            fields = group["features"] + (
                [group["weight_feature"]] if "weight_feature" in group else []
            )
            for field in fields:
                values = np.zeros((count, cap), dtype=self.dtypes[field])
                for positions, indices, block in blocks:
                    selected = block[source][field][indices]
                    width = min(cap, selected.shape[1])
                    values[positions, :width] = selected[:, :width]
                target[field] = values
        check_memory()
        return result

    @contextmanager
    def batches(self, source):
        def encoded():
            try:
                for batch in source:
                    yield self.encode(batch)
            finally:
                close_iterator(source)

        with prefetch(encoded(), self.options["prefetch_batches"]) as stream:
            yield stream


def close_iterator(source):
    close = getattr(source, "close", None)
    if close is not None:
        close()


@contextmanager
def prefetch(source, capacity):
    """FIFO delivery with explicit cancellation; no model/TF calls in the worker."""
    if capacity == 0:
        try:
            yield source
        finally:
            close_iterator(source)
        return
    queue = Queue(maxsize=capacity)
    stopped = Event()
    timer = timing.current()

    def put(value):
        while not stopped.is_set():
            try:
                queue.put(value, timeout=0.1)
                return
            except Full:
                pass

    def produce():
        try:
            for value in source:
                if stopped.is_set():
                    break
                check_memory()
                put(("value", value))
            put(("end", None))
        except BaseException as error:
            put(("error", error))
        finally:
            close_iterator(source)

    worker = Thread(
        target=copy_context().run, args=(produce,), name="tf-input-prefetch", daemon=True
    )
    worker.start()

    def consume():
        while True:
            with timer.span("prefetch_wait"):
                try:
                    kind, value = queue.get(timeout=0.1)
                except Empty:
                    if not worker.is_alive():
                        raise RuntimeError("Input worker ended without terminal record") from None
                    continue
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value

    try:
        yield consume()
    finally:
        stopped.set()
        worker.join()
