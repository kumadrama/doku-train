from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator, Sequence
from typing import BinaryIO, cast

import pyarrow as pa
import pyarrow.parquet as pq

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, ShardManifest
from recommend.train.comm.datasvr.storage import ObjectStore


class ParquetDataset:
    """Replayable shard-at-a-time Parquet reader with content verification."""

    def __init__(
        self,
        store: ObjectStore,
        manifest: DatasetManifest,
        *,
        batch_size: int,
        columns: Sequence[str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._store = store
        self._manifest = manifest
        self._batch_size = batch_size
        self._columns = columns

    def iter_batches(
        self, shards: Iterable[ShardManifest] | None = None
    ) -> Iterator[pa.RecordBatch]:
        selected = tuple(shards) if shards is not None else self._manifest.shards
        for shard in sorted(selected, key=lambda item: (item.event_date, item.uri)):
            yield from self._iter_shard(shard)

    def _iter_shard(self, shard: ShardManifest) -> Iterator[pa.RecordBatch]:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as temporary:
            head = self._store.download_to(shard.uri, cast(BinaryIO, temporary))
            temporary.flush()
            if head.size_bytes != shard.size_bytes or head.sha256 != shard.sha256:
                raise ValueError(f"DATA_INTEGRITY_MISMATCH: {shard.uri}")
            try:
                parquet = pq.ParquetFile(temporary.name)
                if parquet.metadata.num_rows != shard.row_count:
                    raise ValueError(f"DATA_INTEGRITY_MISMATCH: row count for {shard.uri}")
                yield from parquet.iter_batches(
                    batch_size=self._batch_size,
                    columns=self._columns,
                    use_threads=False,
                )
            except pa.ArrowInvalid as error:
                raise ValueError(f"DATA_INTEGRITY_MISMATCH: invalid Parquet {shard.uri}") from error
