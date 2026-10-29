import tensorflow.compat.v1 as tf


def fm_layer(params, x):
    if len(x) == 2:
        dim = x[0].shape[-1]
        return x[0] * x[1][:, :dim]
    sum_square = tf.square(tf.add_n(x))
    square_sum = tf.add_n(list(map(tf.square, x)))
    return 0.5 * (sum_square - square_sum)
