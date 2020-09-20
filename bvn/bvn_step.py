from bvn.get_permutation import get_permutation
import tensorflow as tf
import numpy as np


def get_permutation_tf(edge_matrix):
    """Calculates the maximum cardinality perfect matching using networkx
    that corresponds to a permutation matrix

    Arguments:

    edge_matrix: tf.Tensor
        a binary matrix that corresponds to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    return tf.numpy_function(
        get_permutation, [edge_matrix], tf.float32)


def bvn_step(matrix):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    matrix: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrix
        found for the remaining values in matrix
    coefficient: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficient
        found for the remaining values in matrix"""

    b, m, n = tf.shape(matrix)[0], tf.shape(matrix)[1], tf.shape(matrix)[2]

    # convert the matrix into an edge matrix of a bipartite graph
    top = tf.concat([tf.zeros([b, m, m]), matrix], axis=2)
    edge_matrix = tf.concat([top, tf.concat([tf.transpose(
        matrix, [0, 2, 1]), tf.zeros([b, n, n])], axis=2)], axis=1)

    # get a permutation matrix whose minimum edge weight is maximum
    permutation = tf.map_fn(get_permutation_tf, edge_matrix)
    permutation.set_shape(matrix.get_shape())
    upper_bound = tf.fill(tf.shape(matrix), np.inf)

    return permutation, tf.reduce_min(tf.where(
        tf.equal(permutation, 0), upper_bound, matrix), axis=[1, 2])
