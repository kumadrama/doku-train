from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.vocab_manager import lookup


@pytest.mark.parametrize("dtype,keys", [("int64", [[17, 0]]), ("string", [["new", ""]])])
def test_empty_vocabulary_keeps_valid_oov_and_masks_padding(dtype, keys):
    params = SimpleNamespace(
        vocabs={"empty": {"dtype": dtype, "keys": []}},
        embed_dim={"*": 2},
        embed_initializer_args={"minval": 1.0, "maxval": 1.0},
    )
    with tf.Graph().as_default(), tf.Session() as session:
        values = tf.constant(keys, dtype=tf.as_dtype(dtype))
        embedded = lookup(values, "empty", params, tf.constant([[True, False]]))
        all_padding = lookup(values, "empty", params, tf.constant([[False, False]]))
        variable = tf.trainable_variables()[0]
        gradient = tf.gradients(tf.reduce_sum(embedded), variable)[0]
        padding_gradient = tf.gradients(tf.reduce_sum(all_padding), variable)[0]
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        result, zeros, indices, padded_indices = session.run(
            [embedded, all_padding, gradient.indices, padding_gradient.indices]
        )
        np.testing.assert_array_equal(result, [[[1, 1], [0, 0]]])
        np.testing.assert_array_equal(zeros, [[[0, 0], [0, 0]]])
        assert indices.tolist() == [0]
        assert padded_indices.tolist() == []
        assert variable.shape.as_list() == [1, 2]
