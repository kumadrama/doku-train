"""Mixed sampling preserves finite populations and explicit summed objective."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.datasvr.prepared import batches, file_checksum, rows
from recommend_tf.train.comm.nn_utils import get_loss


def test_mixed_rows_are_reproducible_and_complete(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    files = []
    for source, count in [(0, 90), (1, 10)]:
        path = tmp_path / "sources" / str(source) / "train" / "part.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"id": list(range(source * 100, source * 100 + count))}), path)
        files.append(
            dict(
                file=f"../sources/{source}/train/part.parquet",
                source_id=source,
                rows=count,
                sha256=file_checksum(path),
            )
        )
    (root / "complete.json").write_text(json.dumps(dict(identity="i", files=files, rows=100)))

    def order(seed):
        return [
            int(row["id"])
            for batch in batches(rows(root, "i", seed=seed), 7, seed, 8, replacement=True)
            for row in batch
        ]

    first = order(42)
    assert first == order(42)
    assert first != order(43)
    assert sorted(first) == list(range(90)) + list(range(100, 110))
    assert any(x >= 100 for x in first[:90])


def test_buffer_handles_empty_and_unshuffled_order():
    assert list(batches([], 4, 1, 8)) == []
    assert [x for batch in batches(range(15), 4, None, 8) for x in batch] == list(range(15))


def test_sum_loss_uses_source_weights_without_division():
    with tf.Graph().as_default(), tf.Session() as sess:
        head = dict(
            loss_weight="weights", loss_type="logloss", predict_fn="sigmoid", loss_reduction="sum"
        )
        loss, _ = get_loss(
            head,
            tf.constant([0.0, 0.0, 0.0]),
            tf.constant([1.0, 0.0, 1.0]),
            {"item": {"weights": tf.constant([1.0, 0.0, 0.1])}},
        )
        assert np.isclose(sess.run(loss), np.log(2) * 1.1)
