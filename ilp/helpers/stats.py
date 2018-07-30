import numpy as np
import os
import shelve
import sklearn.preprocessing as prep
from datetime import datetime
from threading import Thread
from enum import Enum
from queue import Queue

from ilp.constants import EPS_32, EPS_64, STATS_DIR
from ilp.helpers.base_class import BaseClass


class JobType(Enum):
    EVAL = 1
    ONLINE_ITER = 3
    LABELED_SAMPLE = 5
    PRINT_STATS = 6
    LABEL_STREAM = 9
    TRAIN_PRED = 11
    TEST_PRED = 12
    RUNTIME = 13
    POINT_PREDICTION = 14


class StatisticsWorker(BaseClass):

    def __init__(self, config, save_file=None, jobs=None, results=None,
                 name='STAT', isave=1, iprint=1000):

        super(StatisticsWorker, self).__init__(name=name)
        if save_file is None:
            cur_time = datetime.now().strftime('%d%m%Y-%H%M%S')
            dataset_name = config['dataset']['name']
            filename = 'stats_' + cur_time + '_' + str(dataset_name) + '.stat'
            save_file = os.path.join(STATS_DIR, filename)
        elif not save_file.endswith('.stat'):
            save_file = save_file + '.stat'

        os.makedirs(os.path.split(save_file)[0], exist_ok=True)
        self.save_file = save_file
        self.config = config
        self.stats = Statistics()
        if jobs is None:
            jobs = Queue()
        self.jobs = jobs
        if results is None:
            results = Queue()
        self.results = results
        self.thread = Thread(target=self.work)
        self.isave = isave
        self.iprint = iprint

    def start(self):
        self.n_iter_eval = 0
        self.thread.start()

    def save(self):
        prev_file = self.save_file + '_iter_' + str(self.n_iter_eval)
        if os.path.exists(self.save_file):
            os.rename(self.save_file, prev_file)
        with shelve.open(self.save_file, 'c') as shelf:
            shelf['stats'] = self.stats.__dict__
            shelf['config'] = self.config
        if os.path.exists(prev_file):
            os.remove(prev_file)

    def stop(self):
        self.jobs.put_nowait({'job': JobType.PRINT_STATS})
        self.jobs.put(None)
        self.thread.join()
        self.save()

    def work(self):
        while True:
            job = self.jobs.get()

            if job is None:  # End of algorithm
                self.jobs.task_done()
                break

            job_type = job['job']

            if job_type == JobType.EVAL:
                self.stats.evaluate(job['y_est'], job['y_true'])
                self.n_iter_eval += 1
            elif job_type == JobType.LABEL_STREAM:
                self.stats.label_stream_true = job['y_true']
                self.stats.label_stream_mask_observed = job['mask_obs']
            elif job_type == JobType.POINT_PREDICTION:
                f = job['vec']
                h = self.stats.entropy(f)
                self.stats.entropy_point_after.append(h)
                y = job['y']
                self.stats.pred_point_after.append(y)
                self.stats.conf_point_after.append(f.max())
            elif job_type == JobType.ONLINE_ITER:
                self.stats.iter_online_count.append(job['n_in_iter'])
                self.stats.iter_online_duration.append(job['dt'])
            elif job_type == JobType.PRINT_STATS:
                err = self.stats.clf_error_mixed[-1] * 100
                self.logger.info('Classif. Error: {:5.2f}%\n\n'.format(err))
            elif job_type == JobType.TRAIN_PRED:
                self.stats.train_est = job['y_est']
            elif job_type == JobType.TEST_PRED:
                y_pred_knn = job['y_pred_knn']
                self.stats.test_pred_knn.append(y_pred_knn)

                y_pred_lp = job['y_pred_lp']
                self.stats.test_pred_lp.append(y_pred_lp)

                self.stats.test_true = y_test = job['y_true']

                err_knn = np.mean(np.not_equal(y_pred_knn, y_test))
                err_lp = np.mean(np.not_equal(y_pred_lp, y_test))
                self.logger.info('knn test err: {:5.2f}%'.format(err_knn*100))
                self.logger.info('ILP test err: {:5.2f}%'.format(err_lp*100))
                self.stats.test_error_knn.append(err_knn)
                self.stats.test_error_olp.append(err_lp)
            elif job_type == JobType.RUNTIME:
                self.stats.runtime = job['t']

            if self.n_iter_eval % self.isave == 0:
                self.save()

            self.jobs.task_done()


EXCLUDED_METRICS = {'label_stream_true',
                    'label_stream_mask_observed',
                    'n_burn_in',
                    'test_pred_knn', 'test_pred_lp', 'test_true',
                    'train_est', 'runtime',
                    'conf_point_after', 'test_error_knn', 'test_error_olp'}


class Statistics:
    """
    Statistics gathered during learning (training and testing).
    """
    def __init__(self):
        self.iter_online_count = []
        self.iter_online_duration = []
        self.n_invalid_samples = []
        self.invalid_samples_ratio = []
        self.clf_error_mixed = []
        self.clf_error_valid = []
        self.l1_error_mixed = []
        self.l1_error_valid = []
        self.cross_ent_mixed = []
        self.cross_ent_valid = []
        self.entropy_pred_mixed = []
        self.entropy_pred_valid = []
        self.label_stream_true = []
        self.label_stream_mask_observed = []
        self.test_pred_knn = []
        self.test_pred_lp = []
        self.test_true = []
        self.train_est = []
        self.runtime = np.nan
        self.test_error_olp = []
        self.test_error_knn = []
        self.entropy_point_after = []
        self.pred_point_after = []
        self.conf_point_after = []

    def evaluate(self, y_predictions, y_true):
        """Computes statistics for a given set of predictions and the ground truth.

        Args:
            y_predictions (array_like): [u_samples, n_classes] soft class predictions for current unlabeled samples
            y_true (array_like): [u_samples, n_classes] one-hot encoding of the true classes_ of the unlabeled samples

            eps (float): quantity slightly larger than zero to avoid division by zero

        Returns:
            float, average accuracy

        """

        u_samples, n_classes = y_predictions.shape

        # Clip predictions to [0,1]
        eps = EPS_32 if y_predictions.itemsize == 4 else EPS_64
        y_pred_01 = np.clip(y_predictions, eps, 1-eps)
        # Normalize predictions to make them proper distributions
        y_pred = prep.normalize(y_pred_01, copy=False, norm='l1')

        # 0-1 Classification error under valid and invalid points
        y_pred_max = np.argmax(y_pred, axis=1)
        y_true_max = np.argmax(y_true, axis=1)
        fc_err_mixed = self.zero_one_loss(y_pred_max, y_true_max)
        self.clf_error_mixed.append(fc_err_mixed)

        # L1 error under valid and invalid points
        l1_err_mixed = np.mean(self.l1_error(y_pred, y_true))
        self.l1_error_mixed.append(l1_err_mixed)

        # Cross-entropy loss
        crossent_mixed = np.mean(self.cross_entropy(y_true, y_pred))
        self.cross_ent_mixed.append(crossent_mixed)

        # Identify valid points (for which a label has been estimated)
        ind_valid, = np.where(y_pred.sum(axis=1) != 0)
        n_valid = len(ind_valid)
        n_invalid = u_samples - n_valid

        self.n_invalid_samples.append(n_invalid)
        self.invalid_samples_ratio.append(n_invalid / u_samples)

        # Entropy of the predictions
        if n_invalid == 0:
            entropy_pred_mixed = np.mean(self.entropy(y_pred))
            self.entropy_pred_mixed.append(entropy_pred_mixed)
            return

        y_pred_valid = y_pred[ind_valid]
        y_true_valid = y_true[ind_valid]

        # 0-1 Classification error under valid points only
        y_pred_valid_max = y_pred_max[ind_valid]
        y_true_valid_max = y_true_max[ind_valid]
        err_valid_max = self.zero_one_loss(y_pred_valid_max, y_true_valid_max)
        self.clf_error_valid.append(err_valid_max)

        # L1 error under valid points only
        l1_err_valid = np.mean(self.l1_error(y_pred_valid, y_true_valid))
        self.l1_error_valid.append(l1_err_valid)

        # Cross-entropy loss
        ce_valid = np.mean(self.cross_entropy(y_true_valid, y_pred_valid))
        self.cross_ent_valid.append(ce_valid)

        # Entropy of the predictions
        entropy_pred_valid = np.mean(self.entropy(y_pred_valid))
        self.entropy_pred_valid.append(entropy_pred_valid)
        n_total = n_valid + n_invalid
        entropy_pred_mixed = (entropy_pred_valid*n_valid + n_invalid) / n_total
        self.entropy_pred_mixed.append(entropy_pred_mixed)


    @staticmethod
    def zero_one_loss(y_pred, y_true, average=True):
        """

        Args:
            y_pred (array_like):  (n_samples, n_classes)
            y_true (array_like):  (n_samples, n_classes)
            average (bool):     Whether to take the average over all predictions.

        Returns:    The absolute difference for each row. 
                    Note that this will be in [0,2] for p.d.f.s.

        """

        if average:
            return np.mean(np.not_equal(y_pred, y_true))
        else:
            return np.sum(np.not_equal(y_pred, y_true))

    @staticmethod
    def l1_error(y_pred, y_true, norm=True):
        """

        Args:
            y_pred (array_like): An array of probability distributions (usually predictions) with shape (n_distros, n_classes)
            y_true (array_like): An array of probability distributions (usually groundtruth) with shape (n_distros, n_classes)
            norm (bool):    Whether to constrain the L1 error to be in [0,1].

        Returns:    The absolute difference for each row. Note that this will be in [0,2] for pdfs.

        """

        l1_error = np.abs(y_pred - y_true).sum(axis=1)
        if norm:
            l1_error /= 2

        return l1_error

    @staticmethod
    def entropy(p, norm=True):
        """

        Args:
            p (array_like): An array of probability distributions with shape (n_distros, n_classes)
            norm (bool):    Whether to normalize the entropy to constrain it in [0,1]

        Returns:    An array of entropies of the distributions with shape (n_distros,)

        """

        entropy = - (p * np.log(p)).sum(axis=1)
        if norm:
            entropy /= np.log(p.shape[1])

        return entropy

    @staticmethod
    def cross_entropy(p, q, norm=True):
        """

        Args:
            p (array_like): An array of probability distributions (usually groundtruth) with shape (n_distros, n_classes)
            q (array_like): An array of probability distributions (usually predictions) with shape (n_distros, n_classes)
            norm (bool):    Whether to normalize the entropy to constrain it in [0,1]

        Returns:    An array of cross entropies between the groundtruth and the prediction with shape (n_distros,)

        """

        cross_ent = -(p * np.log(q)).sum(axis=1)
        if norm:
            cross_ent /= np.log(p.shape[1])

        return cross_ent


def aggregate_statistics(stats_path, metrics=None, excluded_metrics=None):

    print('Aggregating statistics from {}'.format(stats_path))
    if stats_path.endswith('.stat'):
        list_of_files = [stats_path]
    else:
        list_of_files = [os.path.join(stats_path, f) for f in os.listdir(
                        stats_path) if f.endswith('.stat')]

    stats_runs = []
    random_states = []
    for stats_file in list_of_files:
        with shelve.open(stats_file, 'r') as f:
            stats_runs.append(f['stats'])
            random_states.append(f['config']['options']['random_state'])

    print('\nRandom seeds used: {}\n'.format(random_states))

    if metrics is None:
        metrics = Statistics().__dict__.keys()

    if excluded_metrics is None:
        excluded_metrics = EXCLUDED_METRICS

    stats_mean, stats_std = {}, {}
    stats_run0 = stats_runs[0]

    for metric in metrics:
        if metric in excluded_metrics: continue
        if metric not in stats_run0:
            print('\nMetric {} not found!'.format(metric))
            continue
        metric_lists = [stats[metric] for stats in stats_runs]

        # Make a numpy 2D array to merge the different runs
        metric_runs = np.asarray(metric_lists)
        s = metric_runs.shape
        if len(s) < 2:
            print('No values for metric, skipping.')
            continue
        stats_mean[metric] = np.mean(metric_runs, axis=0)
        stats_std[metric] = np.std(metric_runs, axis=0)

    with shelve.open(list_of_files[0], 'r') as f:
        config = f['config']

    lp_times = stats_mean['iter_online_duration']
    ice = stats_mean['clf_error_mixed'][0] * 100
    fce = stats_mean['clf_error_mixed'][-1] * 100
    print('Avg. LP time/iter: {:.4f}s'.format(np.mean(lp_times)))
    print('Initial classification error: {:.2f}%'.format(ice))
    print('Final classification error: {:.2f}%'.format(fce))

    # Add excluded metrics in the end
    for stats_run in stats_runs:
        for ex_metric in excluded_metrics:
            if ex_metric in stats_run:
                print('Appending excluded metric: {}'.format(ex_metric))
                stats_mean[ex_metric] = stats_run[ex_metric]

    if len(list_of_files) == 1:
        stats_std = None

    return stats_mean, stats_std, config
