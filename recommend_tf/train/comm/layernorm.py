"""Finder affine LayerNorm; unsupported dynamic-embedding path omitted."""

import tensorflow

tf = tensorflow.compat.v1


class LayerNorm:
    def __init__(self, x, axis=-1, epsilon=1e-6, center=True, scale=True, remove_batch_mean=True):
        if axis != -1 or not center or not scale or not remove_batch_mean:
            raise ValueError("Only Finder affine ln is migrated")
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.remove_batch_mean = remove_batch_mean

        dim = x.get_shape().as_list()[-1]

        if self.scale:
            self.gamma = tf.get_variable(
                "gamma",
                shape=[dim],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=True,
            )

        if self.center:
            self.beta = tf.get_variable(
                "beta",
                shape=[dim],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=True,
            )

    def forward(self, x):
        if (not self.scale) and (not self.center) and self.remove_batch_mean and self.axis == -1:
            raise ValueError("Only Finder affine ln is migrated")

        input_shape = x.shape
        ndims = len(input_shape)
        broadcast_shape = [1] * ndims
        broadcast_shape[-1] = input_shape.dims[-1].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and -1 != [ndims - 1]:
                return tf.reshape(v, broadcast_shape)
            return v

        mean, variance = tf.nn.moments(x, self.axis, keepdims=True)
        scale = _broadcast(self.gamma) if self.scale else None
        offset = _broadcast(self.beta) if self.center else None
        outputs = tf.nn.batch_normalization(
            x,
            mean if self.remove_batch_mean else mean * 0.0,
            variance,
            offset=offset,
            scale=scale,
            variance_epsilon=self.epsilon,
        )
        return outputs
