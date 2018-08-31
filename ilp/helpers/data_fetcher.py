import os
import gzip
import zipfile
from urllib import request
import yaml
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ilp.constants import DATA_DIR


CWD = os.path.split(__file__)[0]
DATASET_CONFIG_PATH = os.path.join(CWD, 'datasets.yml')

SUPPORTED_DATASETS = {'mnist', 'usps', 'blobs', 'kitti_features'}
IS_DATASET_STREAM = {'kitti_features': True}


def check_supported_dataset(dataset):

    if dataset not in SUPPORTED_DATASETS:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))

    return True


def fetch_load_data(name):

    print('\nFetching/Loading {}...'.format(name))
    with open(DATASET_CONFIG_PATH, 'r') as f:
        datasets_configs = yaml.load(f)
        if name.upper() not in datasets_configs:
            raise FileNotFoundError('Dataset {} not supported.'.format(name))

    config = datasets_configs[name.upper()]
    name_ = config.get('name', name)
    test_size = config.get('test_size', 0)

    if name_ == 'KITTI_FEATURES':
        X_tr, y_tr, X_te, y_te = fetch_kitti()
    elif name_ == 'USPS':
        X_tr, y_tr, X_te, y_te = fetch_usps()
    elif name_ == 'MNIST':
        X_tr, y_tr, X_te, y_te = fetch_mnist()
        X_tr = X_tr / 255.
        X_te = X_te / 255.
    elif name_ == 'BLOBS':
        X, y = make_classification(n_samples=60)
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)

        if test_size > 0:
            if type(test_size) is int:
                t = test_size
                print('{} has shape {}'.format(name_, X.shape))
                print('Splitting data with test size = {}'.format(test_size))
                X_tr, X_te, y_tr, y_te = X[:-t], X[-t:], y[:-t], y[-t:]
            elif type(test_size) is float:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, stratify=y)
            else:
                raise TypeError('test_size is neither int or float.')

            print('Loaded training set with shape {}'.format(X_tr.shape))
            print('Loaded testing  set with shape {}'.format(X_te.shape))
            return X_tr, y_tr, X_te, y_te
        else:
            print('Loaded {} with {} samples of dimension {}.'
                  .format(name_, X.shape[0], X.shape[1]))
            return X, y, None, None
    else:
        raise NameError('No data set {} found!'.format(name_))

    print('Loaded training data   with shape {}'.format(X_tr.shape))
    print('Loaded training labels with shape {}'.format(y_tr.shape))
    print('Loaded testing  data   with shape {}'.format(X_te.shape))
    print('Loaded testing  labels with shape {}'.format(y_te.shape))
    return X_tr, y_tr, X_te, y_te


def fetch_usps(save_dir=None):

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
    train_file = 'zip.train.gz'
    test_file = 'zip.test.gz'
    save_dir = DATA_DIR if save_dir is None else save_dir

    if not os.path.isdir(save_dir):
        raise NotADirectoryError('{} is not a directory.'.format(save_dir))

    train_source = os.path.join(base_url, train_file)
    test_source = os.path.join(base_url, test_file)

    train_dest = os.path.join(save_dir, train_file)
    test_dest = os.path.join(save_dir, test_file)

    def download_file(source, destination):
        if not os.path.exists(destination):
            print('Downloading from {}...'.format(source))
            f, msg = request.urlretrieve(url=source, filename=destination)
            print('HTTP response: {}'.format(msg))
            return f, msg
        else:
            print('Found dataset in {}!'.format(destination))
            return None

    download_file(train_source, train_dest)
    download_file(test_source, test_dest)

    X_train = np.loadtxt(train_dest)
    y_train, X_train = X_train[:, 0].astype(np.int32), X_train[:, 1:]

    X_test = np.loadtxt(test_dest)
    y_test, X_test = X_test[:, 0].astype(np.int32), X_test[:, 1:]

    return X_train, y_train, X_test, y_test


def fetch_kitti(data_dir=None):

    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, 'kitti_features')

    files = ['kitti_all_train.data',
             'kitti_all_train.labels',
             'kitti_all_test.data',
             'kitti_all_test.labels']

    for file in files:
        if file not in os.listdir(data_dir):
            zip_path = os.path.join(data_dir, 'kitti_features.zip')
            target_path = os.path.dirname(zip_path)
            print("Extracting {} to {}...".format(zip_path, target_path))
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_path)
            print("Done.")
            break

    X_train = np.loadtxt(os.path.join(data_dir, files[0]), np.float64, skiprows=1)
    y_train = np.loadtxt(os.path.join(data_dir, files[1]), np.int32, skiprows=1)
    X_test = np.loadtxt(os.path.join(data_dir, files[2]), np.float64, skiprows=1)
    y_test = np.loadtxt(os.path.join(data_dir, files[3]), np.int32, skiprows=1)

    return X_train, y_train, X_test, y_test


def fetch_mnist(data_dir=None):

    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, 'mnist')

    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    # Create path if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(data_dir):
            request.urlretrieve(url + file, os.path.join(data_dir, file))
            print("Downloaded %s to %s" % (file, data_dir))

    def _images(path):
        """Return flattened images loaded from local file."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), '>B', offset=16)
        return pixels.reshape(-1, 784).astype('float64')

    def _labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), '>B', offset=8)

        return integer_labels

    X_train = _images(os.path.join(data_dir, files[0]))
    y_train = _labels(os.path.join(data_dir, files[1]))
    X_test = _images(os.path.join(data_dir, files[2]))
    y_test = _labels(os.path.join(data_dir, files[3]))

    return X_train, y_train, X_test, y_test
