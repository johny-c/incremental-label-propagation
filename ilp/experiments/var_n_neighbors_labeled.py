import os
import time

from ilp.experiments.base import BaseExperiment
from ilp.helpers.data_fetcher import IS_DATASET_STREAM
from ilp.helpers.params_parse import parse_yaml, experiment_arg_parser
from ilp.constants import CONFIG_DIR


class VarNeighborsLabeled(BaseExperiment):

    def __init__(self, n_neighbors_values, params, n_runs, isave=100):
        super(VarNeighborsLabeled, self).__init__(name='k_L', config=params,
                                                  isave=isave, n_runs=n_runs,
                                                  plot_title=r'Influence of '
                                                             r'$k_l$',
                                                  multi_var=True)
        self.n_neighbors_values = n_neighbors_values

    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):

        params = self.config

        for n_neighbors in self.n_neighbors_values:
            params['graph']['n_neighbors_labeled'] = n_neighbors
            save_dir = os.path.join(self.top_dir,  'k_L_' + str(n_neighbors))
            stats_file = os.path.join(save_dir, 'run_' + str(n_run))
            self.logger.info('\n\nExperiment: {}, k_L = {}, run {}...\n'.
                  format(self.name.upper(), n_neighbors, n_run))
            time.sleep(1)
            self._single_run(X_run, y_run, mask_labeled, n_burn_in,
                             stats_file, seed_run, X_test, y_test)


if __name__ == '__main__':

    parser = experiment_arg_parser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset'].lower()
    config_file = os.path.join(CONFIG_DIR, 'var_k_L.yml')
    config = parse_yaml(config_file)

    # Store dataset info
    config.setdefault('dataset', {})
    config['dataset']['name'] = dataset_name
    config['dataset']['is_stream'] = IS_DATASET_STREAM.get(dataset_name, False)

    N_NEIGHBORS_VALUES = config['graph']['n_neighbors_labeled'].copy()

    experiment = VarNeighborsLabeled(N_NEIGHBORS_VALUES, params=config,
                                     n_runs=args['n_runs'])
    if args['plot'] != '':
        experiment.load_plot(path=args['plot'])
    else:
        experiment.run(dataset_name)