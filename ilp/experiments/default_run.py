import os
from datetime import datetime

from ilp.experiments.base import BaseExperiment
from ilp.helpers.data_fetcher import IS_DATASET_STREAM
from ilp.helpers.params_parse import parse_yaml, experiment_arg_parser
from ilp.constants import CONFIG_DIR


class DefaultRun(BaseExperiment):

    def __init__(self, params, n_runs=1, isave=100):
        super(DefaultRun, self).__init__(name='default_run', config=params,
                                         isave=isave, n_runs=n_runs,
                                         plot_title=r'Default run',
                                         multi_var=False)

    def pre_single_run(self, X_run, y_run, mask_labeled, n_burn_in, seed_run,
                       X_test, y_test, n_run):

        cur_time = datetime.now().strftime('%d%m%Y-%H%M%S')
        stats_path = os.path.join(self.top_dir, 'run_' + cur_time)
        self._single_run(X_run, y_run, mask_labeled, n_burn_in, stats_path,
                         seed_run, X_test, y_test)


if __name__ == '__main__':

    # Parse user input
    parser = experiment_arg_parser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset'].lower()
    config_file = os.path.join(CONFIG_DIR, 'default.yml')
    config = parse_yaml(config_file)

    # Store dataset info
    config.setdefault('dataset', {})
    config['dataset']['name'] = dataset_name
    config['dataset']['is_stream'] = IS_DATASET_STREAM.get(dataset_name, False)

    # Instantiate experiment
    experiment = DefaultRun(params=config, n_runs=args['n_runs'])

    if args['plot'] != '':
        # python3 default_run.py -p latest
        experiment.load_plot(path=args['plot'])
    else:
        # python3 default_run.py -d digits
        experiment.run(dataset_name)
