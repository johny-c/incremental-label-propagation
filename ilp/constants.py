import os
import numpy as np

CWD = os.path.abspath(os.path.split(__file__)[0])
PROJECT_DIR = os.path.split(CWD)[0]

SOURCE_DIR = os.path.join(PROJECT_DIR, 'ilp')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

STATS_DIR = os.path.join(SOURCE_DIR, 'stats_res')
EXPERIMENTS_DIR = os.path.join(SOURCE_DIR, 'experiments')
CONFIG_DIR = os.path.join(EXPERIMENTS_DIR, 'cfg')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'results')
# PLOT_DIR = os.path.join(PROJECT_DIR, 'plot')


EPS_32 = np.spacing(np.float32(0))
EPS_64 = np.spacing(np.float64(0))
