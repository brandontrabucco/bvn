from networkx.algorithms.bipartite.matching import maximum_matching
from networkx import from_numpy_matrix
import numpy as np


def find_threshold(valid_matches, weights, edge_matrix):
    """Calculates the largest edge threshold using binary search such that
    the found maximum cardinality matching is a perfect matching

    Arguments:

    valid_matches: dict
        the last found perfect matching of largest edge weight threshold
        found using a binary search
    weights: np.ndarray
        an array of candidate thresholds for the edge weights in sorted order,
        where the largest elements are first
    edge_matrix: tf.Tensor
        a matrix of edge weights that correspond to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    # calculate the current loc of binary search
    n, loc = edge_matrix.shape[1] // 2, (weights.size - 1) // 2

    # calculate the bipartite graph whose edges all have weight of at
    # least the largest threshold found so far
    threshold = weights[loc]
    bipartite_matrix = np.where(edge_matrix >= threshold, 1, 0)

    # calculate the maximum matching using the hopcroft karp algorithm
    matches = maximum_matching(from_numpy_matrix(bipartite_matrix), range(n))
    matches = {u: v % n for u, v in matches.items() if u < n}

    # calculate if the found matching is a perfect matching
    is_perfect_matching = len(matches) == n
    valid_matches = matches if is_perfect_matching else valid_matches

    # otherwise if the result found is a perfect matching
    # then move onto larger thresholds
    if weights.size > 2 and is_perfect_matching:
        return find_threshold(valid_matches, weights[:loc], edge_matrix)

    # otherwise if the result found is not a perfect matching
    # then move onto smaller thresholds
    elif weights.size > 1 and not is_perfect_matching:
        return find_threshold(valid_matches, weights[loc + 1:], edge_matrix)

    # edge case when no valid permutation is a perfect matching and
    # the decomposition terminates with coefficient zero
    if not valid_matches:
        return np.ones((n, n), dtype=np.float32)

    # at the last iteration of binary search return the best
    # permutation matrix found so far
    permutation = np.zeros((n, n), dtype=np.float32)
    permutation[tuple(zip(*valid_matches.items()))] = 1
    return permutation


def get_permutation(edge_matrix):
    """Calculates the largest edge threshold using binary search such that
    the found maximum cardinality matching is a perfect matching

    Arguments:

    edge_matrix: tf.Tensor
        a matrix of edge weights that correspond to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    # obtain a sorted list of edge weights to perform a binary search over
    # to find the large edge weight threshold such that a perfect matching
    # exists in the graph with edges of weight greater than t
    # https://cstheory.stackexchange.com/questions/32321/
    #     weighted-matching-algorithm-for-minimizing-max-weight
    n = edge_matrix.shape[1] // 2
    weights = np.sort(edge_matrix[np.nonzero(edge_matrix)])[::-1]
    return np.ones((n, n), dtype=np.float32) \
        if weights.size == 0 else find_threshold({}, weights, edge_matrix)
