import os
import time
import numpy as np
from sklearn.utils.random import check_random_state
import shelve

from ilp.experiments.base import BaseExperiment
from ilp.helpers.data_fetcher import fetch_load_data, IS_DATASET_STREAM
from ilp.helpers.params_parse import parse_yaml, experiment_arg_parser
from ilp.constants import CONFIG_DIR
from ilp.helpers.data_flow import split_labels_rest, split_burn_in_rest


class VarSamplesLabeled(BaseExperiment):

    def __init__(self, n_labeled_values, params, n_runs=1, isave=100):
        super(VarSamplesLabeled, self).__init__(name='n_L', config=params,
                                                isave=isave, n_runs=n_runs,
                                                plot_title=r'Influence of '
                                                           r'number of labels',
                                                multi_var=True)
        self.n_labeled_values = n_labeled_values

    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):

        config = self.config

        n_labels = config['data']['n_labels']
        save_dir = os.path.join(self.top_dir, 'n_L_' + str(n_labels))
        stats_file = os.path.join(save_dir, 'run_' + str(n_run))
        print('\n\nExperiment: {}, n_labels = {}, run {}...\n'.
              format(self.name.upper(), n_labels, n_run))
        time.sleep(1)
        self._single_run(X_run, y_run, mask_labeled, n_burn_in,
                         stats_file, seed_run, X_test, y_test)

    def run(self, dataset_name, random_state=42):

        config = self.config
        X_train, y_train, X_test, y_test = fetch_load_data(dataset_name)

        n_classes = len(np.unique(y_train))

        # if dataset_name == 'usps':
        #     X_train = np.concatenate((X_train, X_test))
        #     y_train = np.concatenate((y_train, y_test))

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
                n_labeled_burn_in = sum(mask_labeled_burn_in)
                X_burn_in, y_burn_in = X_train[ind_burn_in], \
                                       y_train[ind_burn_in]
                mask_rest = np.ones(len(X_train), dtype=bool)
                mask_rest[ind_burn_in] = False
                X_rest, y_rest = X_train[mask_rest], y_train[mask_rest]

                for nlpc in self.n_labeled_values:
                    n_labels = nlpc*n_classes
                    config['data']['n_labels'] = n_labels

                    rl = (n_labels - n_labeled_burn_in) / len(y_rest)
                    assert rl >= 0
                    mask_labeled_rest = split_labels_rest(y_rest, batch_size=0,
                        seed=seed_run, shuffle=True, ratio_labeled=rl)

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


def save_results(path, results):
    with shelve.open(path, 'c') as shelf:
        shelf['results'] = results


if __name__ == '__main__':
    parser = experiment_arg_parser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset'].lower()
    config_file = os.path.join(CONFIG_DIR, 'var_n_L.yml')
    config = parse_yaml(config_file)

    # Store dataset info
    config.setdefault('dataset', {})
    config['dataset']['name'] = dataset_name
    config['dataset']['is_stream'] = IS_DATASET_STREAM.get(dataset_name, False)

    N_LABELED_PER_CLASS = config['data']['n_labeled_per_class'].copy()

    experiment = VarSamplesLabeled(N_LABELED_PER_CLASS, params=config,
                                   n_runs=args['n_runs'])
    if args['plot'] != '':
        experiment.load_plot(path=args['plot'])
    else:
        experiment.run(dataset_name)
