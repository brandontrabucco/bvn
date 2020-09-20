from bvn.bvn_step import bvn_step
import tensorflow as tf
import numpy as np


TOLERANCE = np.finfo(np.float).eps * 10.


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=None, dtype=tf.int32)])
def bvn(x, max_iterations):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    x: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]
    max_iterations: int
        the maximum number of matrices to compose to reconstruct
        the doubly stochastic matrix x

    Returns:

    permutations: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition
        shapes like [batch_dim, num_permutations, sequence_len, sequence_len]
    coefficients: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficients
        found using the Berkhoff-Von-Neumann decomposition
        shapes like [batch_dim, num_permutations]"""
    b, n = tf.shape(x)[0], tf.cast(tf.shape(x)[2], tf.float32)
    x = x * n

    # keep track of a sequence of all permutations and coefficients
    coefficients = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    permutations = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    j = tf.constant(-1)
    d = tf.reduce_all(tf.equal(x, 0), axis=[1, 2])

    # all permutations with coefficient 0 are set to the identity matrix
    eye_matrix = tf.eye(tf.shape(x)[2], batch_shape=[b])

    while tf.logical_and(tf.logical_not(
            tf.reduce_all(d)), tf.less(j + 1, max_iterations)):
        j = j + 1

        # compute the permutation matrix whose coefficient is maximum
        # we are done if the coefficient is zero
        p, c = bvn_step(x)
        d = tf.logical_or(d, tf.equal(c, tf.zeros_like(c)))

        # when we are done set the permutation to the identity matrix and
        # the coefficient to zero
        p = tf.where(d[:, tf.newaxis, tf.newaxis], eye_matrix, p)
        c = tf.where(d, tf.zeros_like(c), c)

        # iteratively subtract from the source matrix x until that matrix
        # is approximately zero everywhere
        x = x - c[:, tf.newaxis, tf.newaxis] * p
        x = tf.where(tf.less(tf.abs(x), TOLERANCE), tf.zeros_like(x), x)
        d = tf.logical_or(d, tf.reduce_all(tf.equal(x, 0), axis=[1, 2]))

        permutations = permutations.write(j, p)
        coefficients = coefficients.write(j, c)

    # the num_permutations axis is first and needs to be transposed
    return (tf.transpose(permutations.stack(), [1, 0, 2, 3]),
            tf.transpose(coefficients.stack(), [1, 0]) / n)
