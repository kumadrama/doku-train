"""Fixed train-only key mapping; local replacement for Finder private storage."""

import tensorflow.compat.v1 as tf

from .graph_helper import SPARSE_VARIABLES


def lookup(keys, name, params, valid=None):
    if valid is not None:
        # Lookup only real positions: padded positions must not decay the OOV row
        # through a zero-valued IndexedSlices entry.
        selected = lookup(tf.boolean_mask(keys, valid), name, params)
        dim = params.embed_dim.get(name, params.embed_dim["*"])
        shape = tf.concat([tf.shape(keys, out_type=tf.int64), [dim]], axis=0)
        result = tf.scatter_nd(tf.where(valid), selected, shape)
        result.set_shape(keys.shape.concatenate([dim]))
        return result
    state = params.vocabs[name]
    key_dtype = tf.as_dtype(state["dtype"])
    if keys.dtype != key_dtype:
        raise ValueError("Vocabulary key dtype mismatch: " + name)
    with tf.variable_scope("local_embeddings/" + name, reuse=tf.AUTO_REUSE):
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(state["keys"], dtype=key_dtype),
                tf.range(2, len(state["keys"]) + 2, dtype=tf.int64),
            ),
            default_value=1,
        )
        ids = table.lookup(keys)
        dim = params.embed_dim.get(name, params.embed_dim["*"])
        variable = tf.get_variable(
            "weights",
            shape=[len(state["keys"]) + 1, dim],
            initializer=tf.random_uniform_initializer(**params.embed_initializer_args),
        )
        if not any(v is variable for v in tf.get_collection(SPARSE_VARIABLES)):
            tf.add_to_collection(SPARSE_VARIABLES, variable)
        # No trainable padding row: inputs carry true length, padding masked downstream.
        return tf.nn.embedding_lookup(variable, ids - 1)
