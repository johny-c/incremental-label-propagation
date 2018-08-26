import numpy as np
from time import time
from sklearn.utils.extmath import safe_sparse_dot as ssdot
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import normalize

from ilp.algo.knn_graph_utils import construct_weight_mat
from ilp.algo.knn_sl_graph import KnnSemiLabeledGraph
from ilp.helpers.stats import JobType
from ilp.helpers.log import make_logger


class IncrementalLabelPropagation:
    """
    Parameters
    ----------

    kernel : string, optional
        The kind of kernel to use during graph construction.
        Can be either 'knn' or 'rbf'. (default: 'knn')

    gamma : float, optional
        If 'rbf' kernel is used, this is equivalent to the inverse of the
        length scale. (1/sigma ** 2)

    theta : float, optional
        The threshold of significance for a label update (defines the online nature of the algorithm). (default: 0.1)

    max_iter : int, optional
        The maximum number of iterations per point insertion. (default: 30)

    tol : float, optional
        The tolerance of absolute difference between two label distributions
        that indicates convergence. (default: 1e-3)

    params_graph : dict
        Parameters for the graph construction. Will be passed to the Graph
        constructor.

    params_offline_lp : dict
        Parameters for the offline label propagation that will be performed
        once `n_burn_in` number of samples have been observed.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    iprint : integer, optional
        The n_labeled_freq of printing progress messages to stdout.

    isave : integer, optional
        The n_labeled_freq of saving statistics.

    n_jobs : integer, optional
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation. -1 means 'all CPUs'. Defaults to 1.


    Attributes
    ----------

    classes : tuple, shape = (n_classes,)
        The classes expected to appear in the stream.

    """

    def __init__(self, datastore=None, stats_worker=None,
                 params_graph=None, params_offline_lp=None, n_burn_in=0,
                 theta=0.1, max_iter=30, tol=1e-3,
                 random_state=42, n_jobs=-1, iprint=0):

        self.datastore = datastore
        self.max_iter = max_iter
        self.tol = tol
        self.theta = theta
        self.params_graph = params_graph
        self.params_offline_lp = params_offline_lp
        self.n_burn_in = n_burn_in
        self.random_state = random_state

        kernel = params_graph['kernel']
        if kernel == 'knn':
            Graph = KnnSemiLabeledGraph
        # elif kernel == 'm-knn':
        #     Graph = MutualKnnSemiLabeledGraph
        else:
            raise NotImplementedError('Only knn graphs.')

        self.graph = Graph(datastore, **params_graph)

        self.n_jobs = n_jobs
        self.stats_worker = stats_worker
        self.iprint = iprint
        self.logger = make_logger(__name__)

    def fit_burn_in(self):
        """Fit a semi-supervised label propagation model

        A bootstrap data set is provided in matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            A {n_samples by n_samples} size matrix will be created from this

        Returns
        -------
        self : returns an instance of self.
        """

        self.random_state_ = check_random_state(self.random_state)

        # Get appearing classes
        classes = self.datastore.classes

        self.n_iter_online = 0
        self.n_burn_in_ = self.datastore.get_n_samples()
        self.logger.debug('FITTING BURN-IN WITH {} SAMPLES'.format(self.n_burn_in_))

        # actual graph construction (implementations should override this)
        self.logger.debug('Building graph....')
        self.graph.build(self.get_X_l(), self.get_X_u())
        self.logger.debug('Graph built.')

        u_max = self.datastore.max_samples
        # Initialize F_U with uniform label vectors
        self.y_unlabeled = np.full((u_max, len(classes)), 1 / len(classes),
                                   dtype=self.datastore.dtype)

        # Initialize F_U with zero label vectors
        # self.y_unlabeled = np.zeros((u_max, len(classes)),
        #                             dtype=self.datastore.dtype)

        # Offline label propagation on burn-in data set
        self._offline_lp(**self.params_offline_lp)

        # Normalize F_U as it might have numerically diverged from [0, 1]
        normalize(self.y_unlabeled, norm='l1', axis=1, copy=False)

        return self

    def predict(self, X, mode=None):
        """Predict the labels for a batch of data points, without actually 
        inserting them in the graph.

        Parameters
        ----------
        X : array, shape (n_samples_batch, n_features)
            A batch of data points.
        
        mode : string
            Test with nearest labeled neighbors ('knn'), or nearest 
            unlabeled neighbors ('lp'), their combination (default), 
            or return both ('pair').

        Returns
        -------
        y : array, shape (n_samples_batch, n_classes)
            The predicted label distributions for the given points.

        """

        modes = ['knn', 'lp', 'pair']
        if mode is not None and mode not in modes:
            raise ValueError('predict_proba can have modes: {}'.format(modes))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if mode == 'pair':
            y_proba_knn, y_proba_lp = self.predict_proba(X, 'pair')
            y_pred_knn = self.datastore.inverse_transform_labels(y_proba_knn)
            y_pred_lp = self.datastore.inverse_transform_labels(y_proba_lp)
            return y_pred_knn, y_pred_lp

        y_proba = self.predict_proba(X, mode)
        y_pred = self.datastore.inverse_transform_labels(y_proba)

        return y_pred

    def predict_proba(self, X, mode=None):

        modes = ['knn', 'lp', 'pair']
        if mode is not None and mode not in modes:
            raise ValueError('predict_proba can have modes: {}'.format(modes))

        u, l = self.graph.n_unlabeled, self.graph.n_labeled

        self.logger.info('Now testing on {} samples...'.format(len(X)))
        neighbors, distances = self.graph.find_labeled_neighbors(X)
        affinity_mat = construct_weight_mat(neighbors, distances,
                                            (X.shape[0], l), self.graph.dtype)
        p_tl = normalize(affinity_mat.tocsr(), norm='l1', axis=1)
        y_from_labeled = ssdot(p_tl, self.datastore.y_labeled[:l], True)

        neighbors, distances = self.graph.find_unlabeled_neighbors(X)
        affinity_mat = construct_weight_mat(neighbors, distances,
                                            (X.shape[0], u), self.graph.dtype)
        p_tu = normalize(affinity_mat.tocsr(), norm='l1', axis=1)
        y_from_unlabeled = ssdot(p_tu, self.y_unlabeled[:u], True)

        y_pred_proba = y_from_labeled + y_from_unlabeled
        self.logger.info('Labels have been predicted.')

        if mode is None:
            return y_pred_proba
        elif mode == 'knn':
            return y_from_labeled
        elif mode == 'lp':
            return y_from_unlabeled
        elif mode == 'pair':
            return y_from_labeled, y_pred_proba

    def get_X_l(self):
        return self.datastore.get_X_l()

    def get_X_u(self):
        return self.datastore.get_X_u()

    def set_params(self, L):
        self.graph.reset_metric(L)
        tic = time()
        self.graph.build(self.get_X_l(), self.get_X_u())
        toc = time()
        self.logger.info('Reconstructed graph in {:.4f}s\n'.format(toc-tic))

    def _offline_lp(self, return_iter=False, max_iter=30, tol=0.001):
        """Perform the offline label propagation until convergence of the label 
        estimates of the unlabeled points.

        Parameters
        ----------
        return_iter : bool, default=False
            Whether or not to return the number of iterations till convergence 
            of the label estimates.

        Returns
        -------
        y_unlabeled, num_iter : 
            the new label estimates and optionally the number of iterations
        """

        self.logger.debug('Doing Offline LP...')

        u, l = self.graph.n_unlabeled, self.graph.n_labeled

        p_ul = self.graph.subgraph_ul.transition_matrix[:u]
        p_uu = self.graph.subgraph_uu.transition_matrix[:u, :u]
        y_unlabeled = self.y_unlabeled[:u]
        y_labeled = self.datastore.y_labeled

        # First iteration
        y_static = ssdot(p_ul, y_labeled, dense_output=True)

        # Continue loop
        n_iter = 0
        converged = False
        while n_iter < max_iter and not converged:
            y_unlabeled_prev = y_unlabeled.copy()
            y_unlabeled = y_static + ssdot(p_uu, y_unlabeled, True)
            n_iter += 1

            converged = _converged(y_unlabeled, y_unlabeled_prev, tol)

        self.logger.info('Offline LP took {} iterations'.format(n_iter))

        if return_iter:
            return y_unlabeled, n_iter
        else:
            return y_unlabeled

    def fit_incremental(self, x_new, y_new):

        n_samples = self.datastore.get_n_samples()
        if n_samples == 0:
            self.logger.info('\n\nStarting the Statistics Worker\n\n')
            self.stats_worker.start()

        if n_samples < self.n_burn_in:
            self.logger.debug('Still in burn-in phase... observed {:>4} '
                              'points'.format(n_samples))
            self.datastore.append(x_new, y_new)
            if n_samples == self.n_burn_in - 1:
                self.logger.debug('Burn-in complete!')
                self.fit_burn_in()
        else:
            ind_new = self.datastore.append(x_new, y_new)
            self._fit_incremental(x_new, y_new, ind_new)

    def _fit_incremental(self, x_new, y_new, ind_new):
        """Fit a single new point

        Args:
            x_new : array_like, shape (1, n_features) 
                A new data point.
            
            y_new : int
                Label of the new data point (-1 if point is unlabeled).
                
            ind_new : int
                Index of the new point in the data store.

        Returns:
            IncrementalLabelPropagation: a reference to self.
        """

        tic = time()

        if self.n_iter_online == 0:
            self.tic_iprint = time()

        labeled = y_new != -1

        self.graph.add_node(x_new, ind_new, labeled)

        _, n_in_iter = self._propagate_single(ind_new, y_new, return_iter=True)
        self.n_iter_online += 1

        # Update statistics
        dt = time() - tic
        self.log_stats(JobType.ONLINE_ITER, dt=dt, n_in_iter=n_in_iter)

        if not labeled:
            # Track prediction entropy and accuracy
            label_vec = self.y_unlabeled[ind_new][None, :]
            pred = self.datastore.inverse_transform_labels(label_vec)
            self.log_stats(JobType.POINT_PREDICTION, vec=label_vec, y=pred)

        # Print information if needed
        if self.n_iter_online % self.iprint == 0:
            dt = time() - self.tic_iprint
            max_samples = self.datastore.max_samples
            n_samples_curr = self.graph.get_n_nodes()
            n_samples_prev = n_samples_curr - self.iprint
            self.logger.info('Iterations {} to {}/{} took {:.4f}s'.
                              format(n_samples_prev, n_samples_curr,
                                     max_samples, dt))
            self.tic_iprint = time()
            self.log_stats(JobType.PRINT_STATS)

        # Normalize y_u as it might have diverged from [0, 1]
        u = self.graph.n_unlabeled
        normalize(self.y_unlabeled[:u], norm='l1', axis=1, copy=False)

        return self

    def log_stats(self, job_type, **kwargs):
        d = dict(job_type=job_type)
        d.update(**kwargs)
        self.stats_worker.send(d)

    def _propagate_single(self, ind_new, y_new, return_iter=False):
        """Perform label propagation until convergence of the label
        estimates of the unlabeled points. Assume the new node has already 
        been added to the graph, but no label has been estimated.

        Parameters
        ----------
        ind_new : int 
            The index of the new observation determined during graph addition.

        y_new : int 
            The label of the new observation (-1 if point is unlabeled).

        return_iter : bool, default=False
            Whether to return the number of iterations until convergence of 
            the label estimates.

        Returns
        -------
        y_unlabeled, num_iter : returns the new label estimates and optionally 
                                the number of iterations
        """
        # The number of labeled and unlabeled nodes now includes the new point
        y_u = self.y_unlabeled
        y_l = self.datastore.y_labeled

        p_ul = self.graph.subgraph_ul.transition_matrix
        p_uu = self.graph.subgraph_uu.transition_matrix

        a_rev_ul = self.graph.subgraph_ul.rev_adj
        a_rev_uu = self.graph.subgraph_uu.rev_adj

        if y_new == -1:
            # Estimate the label of the new unlabeled point
            label_new = ssdot(p_ul[ind_new], y_l, True) \
                        + ssdot(p_uu[ind_new], y_u, True)
            y_u[ind_new] = label_new

            # The first LP candidates are the unlabeled samples that have
            # the new point as a nearest neighbor
            candidates = a_rev_uu.get(ind_new, set())
        else:
            # The label of the new labeled point is already in the data store
            candidates = a_rev_ul.get(ind_new, set())

        # Initialize a tentative label matrix / hash-map
        y_u_tent = {}  # y_u[:u].copy()

        # Tentative labels are the label est. after the new point insertion
        candid1_norms = []
        for ind in candidates:
            y_u_tent.setdefault(ind, y_u[ind].copy())
            label = ssdot(p_ul[ind], y_l, True) + ssdot(p_uu[ind], y_u, True)
            y_u_tent[ind] = label.ravel()

        n_updates_per_iter = []
        n_iter = 0
        k_u = self.graph.n_neighbors_unlabeled
        u = max(self.graph.n_unlabeled, 1)
        max_iter = int(np.log(u) / np.log(k_u)) if k_u > 1 else self.max_iter
        while len(candidates) and n_iter < max_iter:  # < self.max_iter:

            # Pick the ones that change significantly and change them
            updates, norm = filter_and_update(candidates, y_u_tent, y_u,
                                              self.theta)
            n_updates_per_iter.append(len(updates))

            # Get the next set of candidates (farther from the source)
            candidates = get_next_candidates(updates, y_u_tent, y_u, a_rev_uu,
                                             p_uu)

            n_iter += 1

        # Print the total number of updates
        n_updates = sum(n_updates_per_iter)
        if n_updates:
            self.logger.info('Iter {:6}: {:6} updates in {:2} LP iters, '
                             'max_iter = {:2}'
                .format(self.n_iter_online, n_updates, n_iter, max_iter))

        if return_iter:
            return y_u, n_iter
        else:
            return y_u


def get_next_candidates(major_changes, y_u_tent, y_u, a_rev_uu, p_uu):
    candidates = set()
    for index, label_diff in major_changes:
        back_neighbors = a_rev_uu.get(index, set())
        for neigh in back_neighbors:
            y_u_tent.setdefault(neigh, y_u[neigh].copy())
            y_u_tent[neigh] += ssdot(p_uu[neigh, index], label_diff, True)
            candidates.add(neigh)
    return candidates


def filter_and_update(candidates, y_u_tent, y_u, theta, top_ratio=None):

    # Store for visualization all norms to see how to tune theta
    major_updates = []
    updates_norm = 0.
    candidate_changes = []
    for candidate in candidates:
        dy_u = y_u_tent[candidate] - y_u[candidate]
        dy_u_norm = np.abs(dy_u).sum()
        if top_ratio is not None:
            candidate_changes.append((dy_u_norm, dy_u, candidate))
        else:
            if dy_u_norm > theta:
                # Apply the update
                y_u[candidate] = y_u_tent[candidate]
                major_updates.append((candidate, dy_u))
                updates_norm += dy_u_norm

    if top_ratio is None:
        return major_updates, updates_norm

    # Sort changes by descending norm and select the top k candidates for LP
    n_candidates = len(candidates)
    candidate_changes.sort(reverse=True, key=lambda x: x[0])

    # Apply the changes to the top k candidates
    top_k = int(top_ratio * n_candidates)
    for _, dy_u, candidate in candidate_changes[:top_k]:
        # Apply the update
        y_u[candidate] = y_u_tent[candidate]
        major_updates.append((candidate, dy_u))
        updates_norm += np.abs(dy_u).sum()

    return major_updates, updates_norm


def _converged(y_curr, y_prev, tol=0.01):
    """basic convergence check"""
    return np.abs(y_curr - y_prev).max() < tol
