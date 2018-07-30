import numpy as np
from sklearn.preprocessing import LabelBinarizer, label_binarize

from ilp.constants import EPS_32, EPS_64


class SemiLabeledDataStore:

    def __init__(self, max_samples, max_labeled, classes, precision='float32'):

        self.max_samples = max_samples
        self.max_labeled = max_labeled
        self.classes = classes

        self.precision = precision
        self.dtype = np.dtype(precision)
        self.eps = EPS_32 if self.dtype == np.dtype('float32') else EPS_64
        self.n_labeled = 0
        self.n_unlabeled = 0

        self.X_labeled = np.array([])
        self.X_unlabeled = np.array([])
        self.y_labeled = np.array([])

    def _allocate(self, n_features):

        # Allocate memory for data matrices
        self.X_labeled = np.zeros((self.max_labeled, n_features),
                                  dtype=self.dtype)
        self.X_unlabeled = np.zeros((self.max_samples, n_features),
                                    dtype=self.dtype)

        # Allocate memory for label matrix
        self.y_labeled = np.zeros((self.max_labeled, len(self.classes)),
                                  dtype=self.dtype)

    def append(self, x, y):

        if self.get_n_samples() == 0:
            self._allocate(len(x))

        if y == -1:
            ind_new = self.n_unlabeled
            self.X_unlabeled[ind_new] = x
            self.n_unlabeled += 1
        else:
            ind_new = self.n_labeled
            self.X_labeled[ind_new] = x
            self.y_labeled[ind_new] = label_binarize([y], self.classes)
            self.n_labeled += 1

        return ind_new

    def inverse_transform_labels(self, y_proba):
        if not hasattr(self, 'label_binarizer'):
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(self.classes)

        return self.label_binarizer.inverse_transform(y_proba)

    def get_n_samples(self):
        return self.n_labeled + self.n_unlabeled

    def get_X_l(self):
        return self.X_labeled[:self.n_labeled]

    def get_X_u(self):
        return self.X_unlabeled[:self.n_unlabeled]

    def get_y_l(self):
        return self.y_labeled[:self.n_labeled]

    def get_y_l_int(self):
        return np.argmax(self.get_y_l(), axis=1)