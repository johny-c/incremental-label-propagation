import os
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from ilp.constants import CONFIG_DIR
from ilp.helpers.data_fetcher import SUPPORTED_DATASETS


def args_to_dict(args):

    sample_yaml = os.path.join(CONFIG_DIR, 'default.yml')
    with open(sample_yaml, 'r') as f:
        sample_dict = yaml.load(f)

    params_dict = dict(sample_dict)
    for k, v in args.items():
        for section in sample_dict:
            if k in sample_dict[section]:
                params_dict[section][k] = v

    return params_dict


def parse_yaml(config_file):

    sample_yaml = os.path.join(CONFIG_DIR, 'default.yml')
    with open(sample_yaml, 'r') as f:
        default_params = yaml.load(f)

    with open(config_file) as cfg_file:
        params = yaml.load(cfg_file)

    for k, v in default_params.items():
        if k not in params:
            params[k] = default_params[k]

    return params


def print_config(params):
    for section in params:
        print('\n{} PARAMS:'.format(section.upper()))
        if type(params[section]) is dict:
            for k, v in params[section].items():
                print('{}: {}'.format(k, v))
        else:
            print('{}'.format(params[section]))


def experiment_arg_parser():
    arg_parser = ArgumentParser(description="ILP experiment", formatter_class=ArgumentDefaultsHelpFormatter)

    # Dataset
    arg_parser.add_argument(
        '-d', '--dataset', type=str, default='digits',
        help='Load the given dataset.\nSupported datasets are: {}'
             .format(SUPPORTED_DATASETS)
    )

    arg_parser.add_argument(
        '-n', '--n_runs', type=int, default=1,
        help='Number of times to run the experiment with different seeds'
    )

    arg_parser.add_argument(
        '-p', '--plot', type=str, default='',
        help='Plot the latest experiment results from the given directory'
    )

    return arg_parser
