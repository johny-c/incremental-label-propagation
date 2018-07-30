import os
import numpy as np
import time
from datetime import datetime
from sklearn.externals import six
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.random import check_random_state
from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt

from ilp.constants import RESULTS_DIR
from ilp.helpers.data_fetcher import check_supported_dataset, fetch_load_data
from ilp.helpers.stats import StatisticsWorker, aggregate_statistics, JobType
from ilp.plots.plot_stats import plot_curves
from ilp.algo.incremental_label_prop import IncrementalLabelPropagation
from ilp.algo.datastore import SemiLabeledDataStore
from ilp.helpers.data_flow import gen_semilabeled_data, split_labels_rest, split_burn_in_rest
from ilp.helpers.params_parse import print_config


class BaseExperiment(six.with_metaclass(ABCMeta)):

    def __init__(self, name, config, plot_title, multi_var, n_runs, isave=100):
        self.name = name
        self.config = config
        self.precision = config.get('options', {}).get('precision', 'float32')
        self.isave = isave
        self.plot_title = plot_title
        self.multi_var = multi_var
        self.n_runs = n_runs
        self._setup()

    def _setup(self):

        self.dataset = self.config['dataset']['name'].lower()
        check_supported_dataset(self.dataset)
        cur_time = datetime.now().strftime('%d%m%Y-%H%M%S')
        self.class_dir = os.path.join(RESULTS_DIR, self.name)
        instance_dir = self.name + '_' + self.dataset.upper() + '_' + cur_time
        self.top_dir = os.path.join(self.class_dir, instance_dir)

    def run(self, dataset_name, random_state=42):

        config = self.config
        X_train, y_train, X_test, y_test = fetch_load_data(dataset_name)

        for n_run in range(self.n_runs):
            seed_run = random_state * n_run
            print('\n\nRANDOM SEED = {} for data split.'.format(seed_run))
            rng = check_random_state(seed_run)
            if config['dataset']['is_stream']:
                print('Dataset is a stream. Sampling observed labels.')
                # Just randomly sample ratio_labeled samples for mask_labeled
                n_burn_in = config['data']['n_burn_in_stream']
                ratio_labeled = config['data']['stream']['ratio_labeled']
                n_labeled = int(ratio_labeled*len(y_train))
                ind_labeled = rng.choice(len(y_train), n_labeled,
                                         replace=False)
                mask_labeled = np.zeros(len(y_train), dtype=bool)
                mask_labeled[ind_labeled] = True
                X_run, y_run = X_train, y_train
            else:
                burn_in_params = config['data']['burn_in']
                ind_burn_in, mask_labeled_burn_in = \
                    split_burn_in_rest(y_train, shuffle=True, seed=seed_run,
                                   **burn_in_params)
                X_burn_in, y_burn_in = X_train[ind_burn_in], \
                                       y_train[ind_burn_in]
                mask_rest = np.ones(len(X_train), dtype=bool)
                mask_rest[ind_burn_in] = False
                X_rest, y_rest = X_train[mask_rest], y_train[mask_rest]
                stream_params = config['data']['stream']
                mask_labeled_rest = split_labels_rest(
                    y_rest, seed=seed_run, shuffle=True, **stream_params)

                # Shuffle the rest
                indices = np.arange(len(y_rest))
                rng.shuffle(indices)
                X_run = np.concatenate((X_burn_in, X_rest[indices]))
                y_run = np.concatenate((y_burn_in, y_rest[indices]))
                mask_labeled = np.concatenate((mask_labeled_burn_in,
                                               mask_labeled_rest[indices]))
                n_burn_in = len(y_burn_in)

            config['data']['n_burn_in'] = n_burn_in
            config.setdefault('options', {})
            config['options']['random_state'] = seed_run

            self.pre_single_run(X_run, y_run, mask_labeled, n_burn_in,
                                seed_run, X_test, y_test, n_run)

    @abstractmethod
    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):
        raise NotImplementedError('pre_single_run must be overriden!')

    def _single_run(self, X, y, mask_labeled, n_burn_in, stats_file,
                    random_state, X_test=None, y_test=None):

        lb = LabelBinarizer()
        lb.fit(y)
        print('\n\nLABELS SEEN BY LABEL BINARIZER: {}'.format(lb.classes_))

        # Now print configuration for sanity check
        self.config.setdefault('dataset', {})
        self.config['dataset']['classes'] = lb.classes_
        print_config(self.config)

        print('Creating stream generator...')
        stream_generator = gen_semilabeled_data(X, y, mask_labeled)

        print('Creating one-hot groundtruth...')
        y_u_true_int = y[~mask_labeled]
        y_u_true = np.asarray(lb.transform(y_u_true_int), dtype=self.precision)

        print('Initializing learner...')
        datastore_params = {'precision': self.precision,
                            'max_samples': len(y),
                            'max_labeled': sum(mask_labeled),
                            'classes': lb.classes_}
        ilp = self.init_learner(stats_file, datastore_params, random_state,
                                n_burn_in)

        # Iterate through the generated samples and learn
        t_total = time.time()
        print('Now feeding stream . . .')
        for t, x_new, y_new, is_labeled in stream_generator:

            # Pass the new point to the learner
            y_observed = y_new if is_labeled else -1
            ilp.fit_incremental(x_new, y_observed)

            if t > n_burn_in:
                # Compute classification error
                u = ilp.datastore.n_unlabeled
                y_u = ilp.y_unlabeled[:u]
                d = {'job': JobType.EVAL, 'y_est': y_u, 'y_true': y_u_true[:u]}
                ilp.stats_worker.jobs.put_nowait(d)

                # Compute test error every 1000 samples
                if t % 1000 == 0:
                    if X_test is not None:
                        print('Now testing . . .')
                        t_test = time.time()
                        y_pred_knn, y_pred_lp = ilp.predict(X_test, mode='pair')
                        t_test = time.time() - t_test
                        print('Testing finished in {}s'.format(t_test))
                        d = {'job': JobType.TEST_PRED, 'y_pred_knn': y_pred_knn,
                             'y_pred_lp': y_pred_lp, 'y_true': y_test}
                        ilp.stats_worker.jobs.put_nowait(d)

        # Store the true label stream in statistics
        d = {'job': JobType.LABEL_STREAM, 'y_true': y,'mask_obs': mask_labeled}
        ilp.stats_worker.jobs.put_nowait(d)

        print('Reached end of generated data.')
        total_runtime = time.time() - t_total
        print('Total time elapsed: {} s'.format(total_runtime))

        d = {'job': JobType.RUNTIME, 't': total_runtime}
        ilp.stats_worker.jobs.put_nowait(d)

        # Store last predictions in statistics
        u = ilp.datastore.n_unlabeled
        y_u = ilp.y_unlabeled[:u]
        d = {'job': JobType.TRAIN_PRED, 'y_est': y_u, 'y_true': y_u_true[:u]}
        ilp.stats_worker.jobs.put_nowait(d)

        if X_test is not None:
            print('Now testing . . .')
            t_test = time.time()
            y_pred_knn, y_pred_lp = ilp.predict(X_test, mode='pair')
            t_test = time.time() - t_test
            print('Testing finished in {}s'.format(t_test))
            d = {'job': JobType.TEST_PRED, 'y_pred_knn': y_pred_knn,
                 'y_pred_lp': y_pred_lp, 'y_true': y_test}
            ilp.stats_worker.jobs.put_nowait(d)

        ilp.stats_worker.stop()

    def init_learner(self, stats_file, datastore_params, random_state,
                     n_burn_in):

        config = self.config
        ilp_params = dict(params_offline_lp=config['offline_lp'],
                          params_graph=config['graph'],
                          **config['online_lp'])

        # Instantiate a worker thread for statistics
        stats_worker = StatisticsWorker(config=config, isave=self.isave,
                                        save_file=stats_file)

        # Instantiate a datastore for labeled and unlabeled samples
        datastore = SemiLabeledDataStore(**datastore_params)

        # Instantiate the learner
        ilp = IncrementalLabelPropagation(datastore=datastore, stats_worker=stats_worker,
                                          random_state=random_state, n_burn_in=n_burn_in, **ilp_params)

        return ilp

    def load_plot(self, path=None):
        if path is None:
            path = self.top_dir
        elif not os.path.isdir(path):
            # Load and plot the latest experiment
            print('Experiment Class dir: {}'.format(self.class_dir))
            print('Experiment subdirs: {}'.format(os.listdir(self.class_dir)))
            files_in_class = os.listdir(self.class_dir)
            a_files = [os.path.join(self.class_dir, d) for d in files_in_class]
            list_of_dirs = [d for d in a_files if os.path.isdir(d)]
            path = max(list_of_dirs, key=os.path.getctime)

        print('Collecting statistics from {}'.format(path))
        config = None
        if self.multi_var:
            experiment_stats = []
            for variable_dir in os.listdir(path):
                var_value = variable_dir[len(self.name) + 1:]
                experiment_dir = os.path.join(path, variable_dir)
                stats_mean, stats_std, config = aggregate_statistics(
                    experiment_dir)
                experiment_stats.append((self.name, var_value, stats_mean,
                                         stats_std))
        else:
            stats_mean, stats_std, config = aggregate_statistics(path)
            experiment_stats = (stats_mean, stats_std)

        if config is None:
            raise KeyError('No configuration found for {}'.format(path))

        title = self.plot_title
        plot_curves(experiment_stats, config, title=title, path=path)
        plt.show()
