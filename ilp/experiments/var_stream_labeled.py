import os
import numpy as np
from time import sleep
from sklearn.utils.random import check_random_state

from ilp.experiments.base import BaseExperiment
from ilp.helpers.data_fetcher import fetch_load_data, IS_DATASET_STREAM
from ilp.helpers.params_parse import parse_yaml, experiment_arg_parser
from ilp.constants import CONFIG_DIR


class VarStreamLabeled(BaseExperiment):

    def __init__(self, ratio_labeled_values, params, n_runs=1, isave=100):
        super(VarStreamLabeled, self).__init__(name='srl',
                                               config=params,
                                                isave=isave, n_runs=n_runs,
                                                plot_title=r'Influence of ratio of labels',
                                                multi_var=True)
        self.ratio_labeled_values = ratio_labeled_values

    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):

        config = self.config

        ratio_labeled = config['data']['stream']['ratio_labeled']
        save_dir = os.path.join(self.top_dir, 'srl_' + str(ratio_labeled))
        stats_file = os.path.join(save_dir, 'run_' + str(n_run))
        self.logger.info('\n\nExperiment: {}, ratio_labeled = {}, run {}...\n'.
              format(self.name.upper(), ratio_labeled, n_run))
        sleep(1)
        self._single_run(X_run, y_run, mask_labeled, n_burn_in,
                         stats_file, seed_run, X_test, y_test)

    def run(self, dataset_name, random_state=42):

        config = self.config
        X_train, y_train, X_test, y_test = fetch_load_data(dataset_name)

        for n_run in range(self.n_runs):
            seed_run = random_state * n_run
            self.logger.info('\n\nRANDOM SEED = {} for data split.'.format(seed_run))
            rng = check_random_state(seed_run)
            if config['dataset']['is_stream']:
                self.logger.info('Dataset is a stream. Sampling observed labels.')
                # Just randomly sample ratio_labeled samples for mask_labeled
                n_burn_in = config['data']['n_burn_in_stream']

                for ratio_labeled in self.ratio_labeled_values:

                    config['data']['stream']['ratio_labeled'] = ratio_labeled
                    n_labeled = int(ratio_labeled*len(y_train))
                    ind_labeled = rng.choice(len(y_train), n_labeled,
                                             replace=False)
                    mask_labeled = np.zeros(len(y_train), dtype=bool)
                    mask_labeled[ind_labeled] = True
                    X_run, y_run = X_train, y_train

                    config['data']['n_burn_in'] = n_burn_in
                    config.setdefault('options', {})
                    config['options']['random_state'] = seed_run

                    self.pre_single_run(X_run, y_run, mask_labeled, n_burn_in,
                                        seed_run, X_test, y_test, n_run)


if __name__ == '__main__':
    parser = experiment_arg_parser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset'].lower()
    config_file = os.path.join(CONFIG_DIR, 'var_stream_labeled.yml')
    config = parse_yaml(config_file)

    # Store dataset info
    config.setdefault('dataset', {})
    config['dataset']['name'] = dataset_name
    config['dataset']['is_stream'] = IS_DATASET_STREAM.get(dataset_name, False)

    N_RATIO_LABELED = config['data']['stream']['ratio_labeled'].copy()

    experiment = VarStreamLabeled(N_RATIO_LABELED, params=config,
                                  n_runs=args['n_runs'])
    if args['plot'] != '':
        experiment.load_plot(path=args['plot'])
    else:
        experiment.run(dataset_name)
