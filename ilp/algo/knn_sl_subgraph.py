import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.utils.extmath import safe_sparse_dot as ssdot

from ilp.helpers.fc_heap import FixedCapacityHeap as FSH
from ilp.algo.knn_graph_utils import find_nearest_neighbors, get_nearest, construct_weight_mat


class KnnSubGraph:
    def __init__(self, n_neighbors=1, dtype=float, shape=None, **kwargs):

        self.n_neighbors = n_neighbors
        self.dtype = dtype
        self.shape = shape

        self.weight_matrix = csr_matrix(shape, dtype=dtype)
        self.transition_matrix = csr_matrix(shape, dtype=dtype)
        self.radii = np.zeros(shape[0], dtype=dtype)

        self.adj = {}
        self.rev_adj = {}

    def build(self, X1, X2, L=None):

        neigh_ind, dist = find_nearest_neighbors(X1, X2, self.n_neighbors, L)
        weight_matrix = construct_weight_mat(neigh_ind, dist, self.shape,
                                             self.dtype)
        self.weight_matrix = weight_matrix.tocsr()
        self.radii[:len(dist)] = dist[:, self.n_neighbors-1]
        self.adj = self.adj_from_weight_mat(weight_matrix)
        self.rev_adj = self.rev_adj_from_weight_mat(weight_matrix)

    def adj_from_weight_mat(self, weight_mat):
        """Get the non-zero cols for each row and insert in a FSPQ in the
        form (weight, ind)
    
        Args:
            weight_mat (coo_matrix): a weights submatrix
    
        Returns:
            adj_list (dict): a dictionary where keys are indices and values
            are FSHs
    
        """

        # Create a list of adjacent vertices for each node
        print('Creating hashmap for {} nodes...'.format(self.shape[0]))
        adj_list = {i: [] for i in range(self.shape[0])}
        print('Iterating over weightmat.row, col, data...')
        for r, c, w in zip(weight_mat.row, weight_mat.col, weight_mat.data):
            adj_list[r].append((w, c))

        # Convert each list to a FixedCapacityHeap
        print('Converting to FCH')
        for node, neighbors in adj_list.items():
            adj_list[node] = FSH(neighbors, capacity=self.n_neighbors)

        return adj_list

    def rev_adj_from_weight_mat(self, weight_mat):
        # Create a list of adjacent vertices for each node
        # adj_list = {i: set() for i in range(self.shape[1])}
        adj_list = {}
        for r, c, w in zip(weight_mat.row, weight_mat.col, weight_mat.data):
            adj_list.setdefault(c, set()).add(r)
            # adj_list[c].add(r)

        return adj_list

    def update_transitions(self, normalizer):
        self.transition_matrix = ssdot(normalizer, self.weight_matrix)

    def eliminate_zeros(self):
        self.weight_matrix.eliminate_zeros()
        self.transition_matrix.eliminate_zeros()

    def append_row(self, index, distances):

        # Identify the k nearest neighbors
        nearest, dist_nearest = get_nearest(distances, self.n_neighbors)
        nearest = nearest.ravel()

        # Create the new node's adjacency list
        weights = np.exp(-dist_nearest.ravel())
        lst = [(w, i) for w, i in zip(weights, nearest)]
        self.adj[index] = FSH(lst, self.n_neighbors)

        # Update the reverse adjacency list
        for w, i in zip(weights, nearest):
            self.rev_adj.setdefault(i, set()).add(index)
            # self.rev_adj[i].add(index)

        # Update the W_LL matrix (append the row vector)
        row = [index] * len(weights)
        row_new = csr_matrix((weights, (row, nearest)), self.shape, self.dtype)
        self.weight_matrix = self.weight_matrix + row_new

    def update_columns(self, ind_new, distances):
        """
        
        Parameters
        ----------
        ind_new : int
            Index of the new point.
            
        distances : array
            Array of distances of the new point to the reference points of 
            the subgraph.

        """

        distances = distances.ravel()
        # Identify the samples that have the new point in their knn radius
        back_refs, = np.where(distances < self.radii[:len(distances)])
        back_weights = np.exp(-distances[back_refs])

        # Update the W_LL matrix (compute the column update)
        update_mat = self._update(ind_new, back_refs, back_weights)
        self.weight_matrix = self.weight_matrix + update_mat.tocsr()

    def _update(self, ind_new, back_refs, weights_new, eps=1e-12):
        row, col, val = [], [], []
        # row_del, col_del, val_del = [], [], []
        for neigh_new, weight_new in zip(back_refs, weights_new):
            neighbors_heap = self.adj[neigh_new]  # FSH with (weight, ind)
            inserted, removed = neighbors_heap.push((weight_new, ind_new))
            if inserted:  # neigh got a new nearest neighbor: ind_new
                row.append(neigh_new)
                col.append(ind_new)
                val.append(weight_new)

                # Update the reverse adjacency list
                self.rev_adj.setdefault(ind_new, set()).add(neigh_new)
                if removed is not None:  # old point swapped a nearest neighbor
                    # row_del.append(neigh)
                    # col_del.append(removed[1])
                    # val_del.append(-removed[0])
                    row.append(neigh_new)
                    col.append(removed[1])
                    val.append(-removed[0])

                    # Update the reverse adjacency list
                    self.rev_adj[removed[1]].discard(neigh_new)

                    # Update the radii
                    min_weight = neighbors_heap.get_min()[0]
                    self.radii[neigh_new] = -np.log(max(min_weight, eps))

        update_mat = coo_matrix((val, (row, col)), self.shape, self.dtype)

        return update_mat
