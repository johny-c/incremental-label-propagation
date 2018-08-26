import os
from time import sleep

from ilp.experiments.base import BaseExperiment
from ilp.helpers.data_fetcher import IS_DATASET_STREAM
from ilp.helpers.params_parse import parse_yaml, experiment_arg_parser
from ilp.constants import CONFIG_DIR


class VarTheta(BaseExperiment):

    def __init__(self, theta_values, params, n_runs, isave=100):
        super(VarTheta, self).__init__(name='theta', config=params,
                                       isave=isave, n_runs=n_runs,
                                       plot_title=r'Influence of $\vartheta$',
                                       multi_var=True)
        self.theta_values = theta_values

    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):

        params = self.config

        for theta in self.theta_values:
            params['online_lp']['theta'] = theta
            save_dir = os.path.join(self.top_dir,  'theta_' + str(theta))
            stats_file = os.path.join(save_dir, 'run_' + str(n_run))
            self.logger.info('\n\nExperiment: {}, theta = {}, run {}...\n'.
                  format(self.name.upper(), theta, n_run))
            sleep(1)
            self._single_run(X_run, y_run, mask_labeled, n_burn_in,
                             stats_file, seed_run, X_test, y_test)


if __name__ == '__main__':

    parser = experiment_arg_parser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset'].lower()
    config_file = os.path.join(CONFIG_DIR, 'var_theta.yml')
    config = parse_yaml(config_file)

    # Store dataset info
    config.setdefault('dataset', {})
    config['dataset']['name'] = dataset_name
    config['dataset']['is_stream'] = IS_DATASET_STREAM.get(dataset_name, False)

    THETA_VALUES = config['online_lp']['theta'].copy()

    # Instantiate experiment
    experiment = VarTheta(theta_values=THETA_VALUES, params=config,
                          n_runs=args['n_runs'])
    if args['plot'] != '':
        experiment.load_plot(path=args['plot'])
    else:
        # python3 default_run.py -d digits
        experiment.run(dataset_name)