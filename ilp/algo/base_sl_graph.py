from sklearn.externals import six
from abc import ABCMeta, abstractmethod


class BaseSemiLabeledGraph(six.with_metaclass(ABCMeta)):
    """
    Parameters
    ----------

    datastore : algo.datastore.SemoLabeledDatastore
        A datastore to store observations as they arrive

    max_samples : int, optional (default=1000)
        The maximum number of points expected to be observed. Useful for 
        memory allocation. 

    max_labeled : {float, int}, optional
        Maximum expected labeled points ratio, or number of labeled points

    dtype : dtype, optional (default=np.float32)
        Precision in floats, (can also be float16, float64)

    """

    def __init__(self, datastore):

        self.datastore = datastore
        self.max_samples = datastore.max_samples
        self.max_labeled = datastore.max_labeled
        self.dtype = datastore.dtype
        self.eps = datastore.eps

        self.n_labeled = 0
        self.n_unlabeled = 0


    @abstractmethod
    def build(self, X_l, X_u):
        raise NotImplementedError('build is not implemented!')

    @abstractmethod
    def add_node(self, x, ind, labeled):
        raise NotImplementedError('add_node is not implemented!')

    @abstractmethod
    def add_labeled_node(self, x, ind):
        raise NotImplementedError('add_labeled_node is not implemented!')

    @abstractmethod
    def add_unlabeled_node(self, x, ind):
        raise NotImplementedError('add_unlabeled_node is not implemented!')

    def get_n_nodes(self):
        return self.n_labeled + self.n_unlabeled
