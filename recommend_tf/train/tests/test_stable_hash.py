"""Sampling must not inherit structured account-ID low-bit bias."""

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.common_utils import eval_expr, stable_hash


def hashes(values, dtype):
    with tf.Graph().as_default(), tf.Session() as sess:
        return sess.run(stable_hash(tf.constant(values, dtype=dtype)))


def test_hash_preserves_int64_identity_and_batch_independence():
    ids = [0, -1, 2**53, 2**53 + 1, 2**63 - 1, -(2**63)]
    expected = hashes(ids, tf.int64)
    np.testing.assert_array_equal(
        expected, [740772398, 1198469153, 2074316794, 1124116163, 2067771217, 345667358]
    )
    assert len(set(expected)) == len(ids)
    np.testing.assert_array_equal(expected, hashes([str(x) for x in ids], tf.string))
    np.testing.assert_array_equal(
        expected, np.concatenate([hashes(ids[:3], tf.int64), hashes(ids[3:], tf.int64)])
    )


def test_hash_sampling_accepts_structured_even_ids_and_keeps_idea():
    ids = np.arange(100000, dtype=np.int64) * 8192 + 2**53
    assert not np.any(ids % 50 == 1)
    with tf.Graph().as_default(), tf.Session() as sess:
        values = {"account": tf.constant(ids), "source": tf.constant([0] * len(ids))}
        keep = sess.run(eval_expr("source == 1 or stable_hash(account) % 50 == 1", values))
        values["source"] = tf.ones_like(values["source"])
        assert sess.run(eval_expr("source == 1 or stable_hash(account) % 50 == 1", values)).all()
    assert 1500 < np.count_nonzero(keep) < 2500


def test_hash_rejects_lossy_float_ids():
    with pytest.raises(ValueError, match="integer or string"):
        stable_hash(tf.constant([float(2**53)]))
