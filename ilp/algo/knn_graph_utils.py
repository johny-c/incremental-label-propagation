import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import coo_matrix


def squared_distances(X1, X2, L=None):

    if L is None:
        dist = euclidean_distances(X1, X2, squared=True)
    else:
        dist = euclidean_distances(X1.dot(L.T), X2.dot(L.T), squared=True)

    return dist


def get_nearest(distances, n_neighbors):

    n, m = distances.shape

    neighbors = np.argpartition(distances, n_neighbors - 1, axis=1)
    neighbors = neighbors[:, :n_neighbors]

    return neighbors, distances[np.arange(n)[:, None], neighbors]


def find_nearest_neighbors(X1, X2, n_neighbors, L=None):
    """
    Args:
        X1 (array_like): [n_samples, n_features] input data points
        X2 (array_like): [m_samples, n_features] reference data points
        n_neighbors (int): number of nearest neighbors to find
        L (array) : linear transformation for Mahalanobis distance computation

    Returns:
        tuple:
            (array_like): [n_samples, k_samples] indices of nearest neighbors
            (array_like): [n_samples, k_distances] distances to nearest neighbors

    """

    dist = squared_distances(X1, X2, L)

    if X1 is X2:
        np.fill_diagonal(dist, np.inf)

    n, m = X1.shape[0], X2.shape[0]

    neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
    neigh_ind = neigh_ind[:, :n_neighbors]

    return neigh_ind, dist[np.arange(n)[:, None], neigh_ind]


def construct_weight_mat(neighbors, distances, shape, dtype):

    n, k = neighbors.shape
    rows = np.repeat(range(n), k)
    cols = neighbors.ravel()
    weights = np.exp(-distances.ravel())
    mat = coo_matrix((weights, (rows, cols)), shape, dtype)

    return mat
