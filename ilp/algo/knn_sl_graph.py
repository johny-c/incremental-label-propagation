import numpy as np
from scipy.sparse import spdiags

from ilp.algo.knn_graph_utils import squared_distances, find_nearest_neighbors
from ilp.algo.knn_sl_subgraph import KnnSubGraph
from ilp.algo.base_sl_graph import BaseSemiLabeledGraph

N_ITER_ELIMINATE_ZEROS = 20


class KnnSemiLabeledGraph(BaseSemiLabeledGraph):
    """
    Parameters
    ----------

    n_neighbors_labeled : int, optional (default=1)
        The number of labeled neighbors to use if 'knn' kernel us used.

    n_neighbors_unlabeled : int, optional (default=7)
        The number of labeled neighbors to use if 'knn' kernel us used.

    max_samples : int, optional
        The maximum number of points expected to be observed. Useful for
        memory allocation. (default: 1000)

    max_labeled : {float, int}, optional
        Maximum expected labeled points ratio, or number of labeled points

    dtype : dtype, optional
        Precision in floats, (default is float32, can also be float16, float64)


    Attributes
    ----------

    L : array-like, shape (n_features_out, n_features_in)

    weight_matrix_{xy} : {array-like, sparse matrix}, shape = [n_samples, n_samples]
                xy can be in {ll, lu, ul, uu} indicating labeled or unlabeled points

    transition_matrix_{xy} : {array-like, sparse matrix}, shape = [n_samples, n_samples]

    adj_list_{xy} : dict that contains the graph connectivity ->
        keys = nodes indices, values = neighbor nodes indices

    """

    def __init__(self, datastore, n_neighbors_labeled=1,
                 n_neighbors_unlabeled=7, **kwargs):

        super(KnnSemiLabeledGraph, self).__init__(datastore=datastore)
        self.n_neighbors_labeled = n_neighbors_labeled
        self.n_neighbors_unlabeled = n_neighbors_unlabeled

        max_l, max_u = self.max_labeled, self.max_samples
        dtype = self.dtype

        self.subgraph_ll = KnnSubGraph(n_neighbors=n_neighbors_labeled,
                                       dtype=dtype, shape=(max_l, max_l))

        self.subgraph_lu = KnnSubGraph(n_neighbors=n_neighbors_unlabeled,
                                       dtype=dtype, shape=(max_l, max_u))

        self.subgraph_ul = KnnSubGraph(n_neighbors=n_neighbors_labeled,
                                       dtype=dtype, shape=(max_u, max_l))

        self.subgraph_uu = KnnSubGraph(n_neighbors=n_neighbors_unlabeled,
                                       dtype=dtype, shape=(max_u, max_u))

        self.L = None

    def build(self, X_l, X_u):
        """
        Graph matrix for Online Label Propagation computes the weighted adjacency matrix

        Parameters
        ----------

        X_l : array-like, shape [l_samples, n_features], the labeled features

        X_u : array-like, shape [u_samples, n_features], the unlabeled features

        """

        print('Building graph with {} labeled and {} unlabeled '
              'samples...'.format(X_l.shape, X_u.shape))

        self.subgraph_ll.build(X_l, X_l, self.L)
        self.subgraph_lu.build(X_l, X_u, self.L)
        self.subgraph_ul.build(X_u, X_l, self.L)
        self.subgraph_uu.build(X_u, X_u, self.L)

        self.n_labeled = X_l.shape[0]
        self.n_unlabeled = X_u.shape[0]

        print('Computing transitions...')

        self._compute_transitions()

    def find_labeled_neighbors(self, X):

        X_l = self.datastore.X_labeled[:self.n_labeled]
        return find_nearest_neighbors(X, X_l, self.n_neighbors_labeled, self.L)

    def find_unlabeled_neighbors(self, X):

        X_u = self.datastore.X_unlabeled[:self.n_unlabeled]
        return find_nearest_neighbors(X, X_u, self.n_neighbors_unlabeled,
                                      self.L)

    def add_node(self, x, ind, labeled):
        if labeled:
            res = self.add_labeled_node(x, ind)
        else:
            res = self.add_unlabeled_node(x, ind)

        # Periodically remove explicit zeros from the sparse matrices
        if self.get_n_nodes() % N_ITER_ELIMINATE_ZEROS == 0:

            self.subgraph_ll.eliminate_zeros()
            self.subgraph_lu.eliminate_zeros()
            self.subgraph_ul.eliminate_zeros()
            self.subgraph_uu.eliminate_zeros()

        return res

    def add_labeled_node(self, x_new, ind_new):

        # Compute distances to all other labeled nodes
        X_l = self.datastore.X_labeled[:self.n_labeled]
        distances = squared_distances(x_new.reshape(1, -1), X_l, self.L)

        # Update the labeled-labeled subgraph
        self.subgraph_ll.append_row(ind_new, distances)
        self.subgraph_ll.update_columns(ind_new, distances)

        # Compute distances to all other unlabeled nodes
        X_u = self.datastore.X_unlabeled[:self.n_unlabeled]
        distances = squared_distances(x_new.reshape(1, -1), X_u, self.L)

        # Update the labeled-unlabeled subgraph
        self.subgraph_lu.append_row(ind_new, distances)

        # Update the unlabeled-labeled subgraph
        self.subgraph_ul.update_columns(ind_new, distances)

        self.n_labeled += 1

        # Compute normalized weight matrix (matrices)
        self._compute_transitions()

        return ind_new

    def add_unlabeled_node(self, x_new, ind_new):

        # Compute distances to all other labeled nodes
        X_u = self.datastore.X_labeled[:self.n_unlabeled]
        distances = squared_distances(x_new.reshape(1, -1), X_u, self.L)

        # Update the labeled-labeled subgraph
        self.subgraph_uu.append_row(ind_new, distances)
        self.subgraph_uu.update_columns(ind_new, distances)

        # Compute distances to all labeled nodes
        X_l = self.datastore.X_labeled[:self.n_labeled]
        distances = squared_distances(x_new.reshape(1, -1), X_l, self.L)

        # Update the labeled-unlabeled subgraph
        self.subgraph_ul.append_row(ind_new, distances)

        # Update the unlabeled-labeled subgraph
        self.subgraph_lu.update_columns(ind_new, distances)

        self.n_unlabeled += 1

        # Compute normalized weight matrix (matrices)
        self._compute_transitions()

        return ind_new

    def _compute_transitions(self):
        """Normalize the weight matrices by dividing with the row sums"""

        self.row_sum_l = self.subgraph_ll.weight_matrix.sum(axis=1) + \
                         self.subgraph_lu.weight_matrix.sum(axis=1)

        self.row_sum_u = self.subgraph_ul.weight_matrix.sum(axis=1) + \
                         self.subgraph_uu.weight_matrix.sum(axis=1)

        # Avoid division by zero
        actual_l = self.row_sum_l[:self.n_labeled]
        actual_l[actual_l < self.eps] = 1.
        # print('Min value l: ', actual_l.min())
        actual_u = self.row_sum_u[:self.n_unlabeled]
        actual_u[actual_u < self.eps] = 1.
        # print('Min value u: ', actual_u.min())

        row_sum_l_inv = 1 / np.asarray(actual_l, dtype=self.dtype)
        row_sum_l_inv[row_sum_l_inv == np.inf] = 1

        row_sum_u_inv = 1 / np.asarray(actual_u, dtype=self.dtype)
        row_sum_u_inv[row_sum_u_inv == np.inf] = 1

        # Temporary divisors (diagonal pre-multiplier matrices)
        diag_l = spdiags(row_sum_l_inv.ravel(), 0, *self.subgraph_ll.shape)
        diag_u = spdiags(row_sum_u_inv.ravel(), 0, *self.subgraph_uu.shape)

        self.subgraph_ll.update_transitions(diag_l)
        self.subgraph_lu.update_transitions(diag_l)
        self.subgraph_ul.update_transitions(diag_u)
        self.subgraph_uu.update_transitions(diag_u)

    def reset_metric(self, L):
        self.L = L
