import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from itertools import count, product
from math import ceil
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from ilp.helpers.params_parse import print_config
from ilp.helpers.data_fetcher import fetch_load_data


COLORS = ['red', 'darkorange', 'black', 'green', 'cyan', 'blue']

N_AXES_PER_ROW = 3

PLOT_LABELS  = {'iter_online_count': r'\#LP iterations',
                'iter_online_duration': r'LP time (s)',
                'n_invalid_samples' : r'\#Invalid samples',
                'invalid_samples_ratio': r'Invalid samples ratio',
                'clf_error_mixed': r'classification error',
                'l1_error_mixed': r'$\ell_1$ error',
                'cross_ent_mixed': r'cross entropy',
                'entropy_pred_mixed': r'prediction entropy',
                'theta': r'$\vartheta$',
                'k_L': r'$k_l$',
                'k_U': r'$k_u$',
                'n_L': r'\#labels',
                'srl': r'labeled',
                'entropy_point_after': r'$H(f)$',
                'conf_point_after': r'$\max_c f_i$',
                'test_error_ilp': 'ILP test error',
                'test_error_knn': 'test error (knn)'
                }

PLOT_TITLE = {'entropy_point_after': 'Entropy',
              'conf_point_after': 'Confidence'}

METRIC_ORDER = [
                'l1_error_mixed',
                'cross_ent_mixed',
                'clf_error_mixed' ,
                'entropy_point_after',
                'entropy_pred_mixed',
                'iter_online_count',
                'iter_online_duration',
                'test_error_ilp',
                'test_error_knn'
                ]

METRIC_TITLE = {'l1_error_mixed':
                    r'$\frac{1}{2u}\sum\limits_{i} '
                    r'|{F_U}_{(i)}^{True} - {F_U}_{(i)}|_1$',
                'cross_ent_mixed':
                    r'$\frac{1}{u}\sum\limits_{i} H({F_U}_{(i)}^{True}, '
                    r'{F_U}_{(i)})$',
                'entropy_pred_mixed':
                    r'$\frac{1}{u}\sum\limits_{i} H({F_U}_{(i)})$',
                'clf_error_mixed':
                    r'$\frac{1}{u}\sum\limits_{i} I(\arg \max {F_U}_{(i)} '
                    r'\neq \arg \max {F_U}_{(i)}^{True})$'}

SCATTER_METRIC = ['iter_online_duration', 'iter_online_count']

LEGEND_METRIC = 'clf_error_mixed'

DECORATORS = {'iter_offline', 'burn_in_labels_true', 'label_stream_true'}

X_LABEL_DEFAULT = r'\#observed samples'
DEFAULT_COLOR = 'b'
DEFAULT_MEAN_COLOR = 'r'
DEFAULT_STD_COLOR = 'darkorange'
COLOR_MAP = plt.cm.inferno
N_CURVES = 6  # THETA \in [0., 0.4, 0.8, 1.2, 1.6, 2.0]
COLOR_IDX = np.linspace(0, 1, N_CURVES + 2)[1:-1]

KITTI_CLASSES = ['car', 'van', 'truck', 'pedestrian', 'sitter', 'cyclist',
                 'tram', 'misc']


def remove_frame(top=True, bottom=True, left=True, right=True):

    ax = plt.gca()
    if top:
        ax.spines['top'].set_visible(False)

    if bottom:
        ax.spines['bottom'].set_visible(False)

    if left:
        ax.spines['left'].set_visible(False)

    if right:
        ax.spines['right'].set_visible(False)


def print_latex_table(stats_list):

    headers = ['\#Labels', 'Runtime (s)', 'Est. error (%)',
               'knn error (%)', 'ILP error (%)']
    table = []

    for _, var_value, stats_value, _ in stats_list:
        runtime = stats_value['runtime']
        est_err = stats_value['clf_error_mixed'][-1]

        y_pred_knn = stats_value['test_pred_knn']
        y_pred = stats_value['test_pred_lp']
        y_true = stats_value['test_true']
        test_err_knn = np.mean(np.not_equal(y_true, y_pred_knn))
        test_err_ilp = np.mean(np.not_equal(y_true, y_pred))

        runtime = '{:6.2f}'.format(runtime)
        est_err = '{:5.2f}'.format(est_err*100)

        test_err_knn = '{:5.2f}'.format(test_err_knn*100)
        test_err_ilp = '{:5.2f}'.format(test_err_ilp*100)
        row = [var_value, runtime, est_err, test_err_knn, test_err_ilp]
        table.append(row)

    print(tabulate(table, headers, tablefmt="latex"))


def plot_histogram(ax, values, title, xlabel, ylabel, value_range):

    weights = np.ones_like(values) / float(len(values))
    bin_y, bin_x, _ = ax.hist(values, range=value_range, normed=False, bins=20,
                              weights=weights, alpha=0.5, align='mid')
    print('Bin values min/max: {}, {}'.format(bin_y.min(), bin_y.max()))

    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_metric_histogram(stats_value, ax1, ax2, metric_key, pos=1):

    if metric_key not in stats_value:
        print('FOUND NO KEY {}.'.format(metric_key))
        return
    if 'pred_point_after' not in stats_value:
        print('FOUND NO KEY pred_point_after.')
        return

    y_u_metric = np.asarray(stats_value[metric_key])
    print('START PLOT HISTOGRAM FOR {}'.format(metric_key))
    y_u_pred = stats_value['pred_point_after'].ravel()

    n = len(y_u_metric)
    y_true = stats_value['label_stream_true']
    mask_labeled = stats_value['label_stream_mask_observed']
    y_u_true = y_true[~mask_labeled]

    mask_correct = np.equal(y_u_pred, y_u_true[-n:])
    metric_hits = y_u_metric[mask_correct]
    metric_miss = y_u_metric[~mask_correct]

    value_range = (0., 1.)
    ylabel = 'Samples ratio'
    xlabel = PLOT_LABELS[metric_key]

    if pos == 1:
        title = PLOT_TITLE[metric_key] + ' - correct predictions'
        xlabel = ''
    elif pos == -1:
        title = ''
    else:
        xlabel = ''
        title = ''
    plot_histogram(ax1, metric_hits, title, xlabel, ylabel, value_range)

    if pos == 1:
        title = PLOT_TITLE[metric_key] + ' - false predictions'
        xlabel = ''
    elif pos == -1:
        title = ''
    else:
        xlabel = ''
        title = ''
    plot_histogram(ax2, metric_miss, title, xlabel, ylabel, value_range)


def plot_metric_histograms(stats, metric_key):

    if type(stats) is list:
        fig = plt.figure(6, (8, 2.5*len(stats)), dpi=200)
        sp_count = count(1)
        for i, (_, var_value, stats_value, _) in enumerate(stats):
            ax1 = fig.add_subplot(len(stats), 2, next(sp_count))
            ax2 = fig.add_subplot(len(stats), 2, next(sp_count))

            if i == 0:
                pos = 1
            elif i == len(stats) - 1:
                pos = -1
            else:
                pos = 0

            plot_metric_histogram(stats_value, ax1, ax2, metric_key, pos)

            title = r'$\vartheta = $' + str(var_value)
            ax1.set_label(title)
            plt.legend()
    else:
        fig = plt.figure(6, (8, 2.5), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        plot_metric_histogram(stats, ax1, ax2, metric_key)

    fig.subplots_adjust(top=0.9)
    fig.tight_layout()


def plot_class_distro_stream(ax, y_true, mask_observed, n_burn_in, classes):
    """Plot incoming label distributions"""
    print('Plotting {}'.format('labels stream distro'))

    xx = np.arange(len(y_true))
    x_lu = [xx[mask_observed], xx[~mask_observed]]
    y_lu = [y_true[mask_observed], y_true[~mask_observed]]
    c_lu = ['red', 'gray']
    sizes = [2, 1]
    markers = ['d', '.']

    n_labeled = sum(mask_observed)
    n_unlabeled = len(y_true) - n_labeled
    labels = [r'labeled ({})'.format(n_labeled),
              r'unlabeled ({})'.format(n_unlabeled)]

    for x, y, c, s, m, label in zip(x_lu, y_lu, c_lu, sizes, markers, labels):
        ax.scatter(x, y, c=c, marker=m, s=s, label=label)

    burn_in_label = 'burn-in ({})'.format(n_burn_in)
    ax.vlines(n_burn_in, *ax.get_ylim(), colors='blue', linestyle=':',
              label=burn_in_label)

    classes_type = type(classes[0])
    if classes_type is str:
        true_labels = np.unique(y_true)
        plt.yticks(range(len(classes)), classes, rotation=45, fontsize=7)
    else:
        ax.set_yticks(classes)

    ax.set_xlabel(X_LABEL_DEFAULT)
    ax.set_ylabel(r'class labels')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc='upper right')


def plot_corrected_samples(y_true, y_pred1, y_pred2, config, n_samples=50):

    dataset = config['dataset']['name']
    _, _, X_test, y_test = fetch_load_data(dataset)

    mask_miss1 = np.not_equal(y_true, y_pred1)
    print('KNN missed {} test cases'.format(sum(mask_miss1)))
    mask_hits2 = np.equal(y_true, y_pred2)
    print('ILP missed {} test cases'.format(sum(~mask_hits2)))
    mask_miss1_hits2 = np.logical_and(mask_miss1, mask_hits2)
    print('ILP missed {} less than KNN'.format(sum(mask_miss1_hits2)))

    samples = X_test[mask_miss1_hits2]
    y_pred_miss = y_pred1[mask_miss1_hits2]
    y_pred_hit = y_pred2[mask_miss1_hits2]
    print('THERE ARE {} CASES OF MISS/HIT'.format(len(samples)))

    if len(samples) > n_samples:
        ind = np.random.choice(len(samples), n_samples, replace=False)
        samples = samples[ind]
        y_pred_miss = y_pred_miss[ind]
        y_pred_hit = y_pred_hit[ind]

    fig = plt.figure(4, figsize=(4, 4), dpi=200)
    dim = int(np.sqrt(len(X_test[0])))

    n_subplots = len(samples)
    n_cols = 10
    n_empty_rows = 0 #  2
    n_rows = ceil(n_subplots / n_cols) + n_empty_rows
    subplot_count = count(1)

    for x, ym, yh in zip(samples, y_pred_miss, y_pred_hit):
        i = next(subplot_count)
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.set_xticks([])
        ax.set_yticks([])

        image = x.reshape(dim, dim)
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title(r'{}$\rightarrow${}'.format(ym, yh), fontsize=8,
                     fontweight='bold', horizontalalignment="center")
        fig.add_subplot(ax)

    fig.suptitle('Corrected samples')


def plot_confusion_matrix(y_true, y_pred, title, cmap=plt.cm.Greys):

    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    true_labels = [str(int(y)) for y in np.unique(y_true)]
    pred_labels = [str(int(y)) for y in np.unique(y_pred)]
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, fontsize=6, fontweight='bold')
    plt.yticks(ytick_marks, pred_labels, fontsize=6, fontweight='bold')

    [i.set_color("b") for i in plt.gca().get_xticklabels()]
    [i.set_color("b") for i in plt.gca().get_yticklabels()]

    thresh = cm_norm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        s = r'{:4}'.format(str(cm[i, j]))
        s = r'\textbf{' + s + r'}'
        plt.text(j, i, s, fontsize=8,
                 horizontalalignment="center", verticalalignment='center',
                 color="white" if cm_norm[i, j] > thresh else "black")

    plt.tick_params(top='off', bottom='off', left='off', right='off',
                    labelleft='on', labelbottom='on')

    remove_frame()

    plt.tight_layout()
    ax = plt.gca()
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)


def plot_confusion_matrices(stats, config, improvements=True):
    matplotlib.rcParams['text.latex.unicode'] = True
    print('Plotting confusion matrices...')

    y_pred_knn = stats['test_pred_knn']
    y_pred_ilp = stats['test_pred_lp']
    y_true = stats['test_true']

    err_knn = np.mean(np.not_equal(y_pred_knn, y_true))
    err_ilp = np.mean(np.not_equal(y_pred_ilp, y_true))
    print('knn error: {}'.format(err_knn))
    print('ilp error: {}'.format(err_ilp))

    fig = plt.figure(3, figsize=(8, 4), dpi=200)
    n_subplots = 2

    fig.add_subplot(1, n_subplots, 1)
    plot_confusion_matrix(y_true, y_pred_knn, '$knn$')

    fig.add_subplot(1, n_subplots, 2)
    plot_confusion_matrix(y_true, y_pred_ilp, 'ILP')

    if improvements:
        plot_corrected_samples(y_true, y_pred_knn, y_pred_ilp,
                               config, n_samples=40)

    dt = config['dataset']['name'].upper()
    fig.suptitle(r'\textbf{Confusion}' + ' (' + dt + ')')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)


def plot_single_run_single_var(stats, fig, config=None):

    metrics_to_plot = []
    for metric_key in METRIC_ORDER:
        if metric_key in stats and len(stats[metric_key]):
            metrics_to_plot.append(metric_key)

    n_subplots = len(metrics_to_plot)
    if 'label_stream_true' in stats:
        n_subplots += 1
    if 'iter_offline_duration' in stats:
        n_subplots += 1

    n_cols = 4
    n_rows = ceil(n_subplots / n_cols)
    plot_count = count(1)

    n = len(stats['iter_online_count'])
    xx = range(n)
    for metric_key in metrics_to_plot:
        print('Plotting metric {}'.format(metric_key))
        ax = fig.add_subplot(n_rows, n_cols, next(plot_count))
        if metric_key in SCATTER_METRIC:
            ax.scatter(xx, stats[metric_key], s=2)
        else:
            ax.plot(stats[metric_key], color=DEFAULT_COLOR, label=None, lw=2)

        ax.set_xlabel(X_LABEL_DEFAULT)
        ax.set_ylabel(PLOT_LABELS[metric_key])

    if 'label_stream_true' in stats and config is not None:
        ax = fig.add_subplot(n_rows, n_cols, next(plot_count))
        y_true = np.asarray(stats['label_stream_true'])
        y_mask = np.asarray(stats['label_stream_mask_observed'])
        n_burn_in = config['data'].get('n_burn_in', 0)
        classes = np.unique(y_true)
        dataset_config = config.get('dataset', None)
        if dataset_config is not None:
            classes = dataset_config.get('classes', classes)
            if dataset_config['name'].startswith('kitti'):
                classes = KITTI_CLASSES
        plot_class_distro_stream(ax, y_true, y_mask, n_burn_in, classes)

    iod = stats.get('iter_offline_duration', None)
    if iod is not None:
        if len(iod):
            ax = fig.add_subplot(n_rows, n_cols, next(plot_count))
            ax.scatter(stats['iter_offline'], iod)
            ax.set_xlabel(X_LABEL_DEFAULT)
            ax.set_ylabel(PLOT_LABELS['iter_offline_duration'])

    fig.subplots_adjust(top=0.9)


def plot_single_run_multi_var(stats_list, fig, config):
    # Single run for each value a variable (e.g. \vartheta) takes.

    kitti = config['dataset']['name'].startswith('kitti')
    LW = 1.5

    var_name, var_value0, stats0, _  = stats_list[0]
    var_label = PLOT_LABELS[var_name]
    metrics_to_plot = []

    metric_order = METRIC_ORDER
    if var_name.startswith('k'):
        metric_order = METRIC_ORDER[:3]
    for metric_key in metric_order:
        print('Metric {}'.format(metric_key))
        print(stats0[metric_key])
        if metric_key in stats0:
            if hasattr(stats0[metric_key], '__len__'):
                if len(stats0[metric_key]):
                    print('Metric to plot: {}'.format(metric_key))
                    metrics_to_plot.append(metric_key)

    stats_list.sort(key=lambda x: float(x[1]))

    for i, (_, var_value, stats_value, _) in enumerate(stats_list):
        rt = stats_value['runtime']
        print('theta = {}, runtime = {}'.format(var_value, rt))

    if len(stats_list) > 7:
        # for n_neighbors pick 1,3,5,7,11,15,19 -> idx = [0,1,2,3,5,7,9]
        stats_list = [stats_list[i] for i in [0,1,2,3,5,7,9]]

    n_subplots = len(metrics_to_plot)
    n_cols = N_AXES_PER_ROW if n_subplots >= N_AXES_PER_ROW else n_subplots
    n_rows = ceil(n_subplots / n_cols)
    plot_count = count(1)

    n_values = len(stats_list)
    if n_values != N_CURVES:
        color_idx = np.linspace(0, 1, n_values+2)
        color_idx = color_idx[1:-1]
    else:
        color_idx = COLOR_IDX

    n = len(stats0['iter_online_count'])
    xx = range(n)
    for metric_key in metrics_to_plot:
        print('Plotting metric {}'.format(metric_key))
        ax = fig.add_subplot(n_rows, n_cols, next(plot_count))
        for i, (_, var_value, stats_value, _) in enumerate(stats_list):
            c = COLOR_MAP(color_idx[i])

            val = int(float(var_value)*100)
            label = r'{}={}\%'.format(var_label, int(val))
            if metric_key in SCATTER_METRIC:
                ax.scatter(xx, stats_value[metric_key], s=1, color=c,
                           label=label)
            else:
                if kitti and metric_key.startswith('test_error'):
                    test_errors = stats_value[metric_key]
                    print('Final Test error: {:5.2f} for {}% of labels'.format(
                        test_errors[-1]*100, var_value))
                    test_times = np.arange(1, len(test_errors) + 1)* 1000
                    ax.plot(test_times, test_errors, color=c, label=label,
                            lw=LW, marker='.')
                    ax.set_ylim((0.0, 0.5))
                    # ax.set_xticks(test_times)
                else:
                    ax.plot(stats_value[metric_key], color=c, label=label,
                            lw=LW)

        ax.set_xlabel(X_LABEL_DEFAULT)
        ax.set_ylabel(PLOT_LABELS[metric_key])
        if metric_key == LEGEND_METRIC:
            plt.legend(loc='best')

    plt.legend(loc='best')


def plot_multi_run_single_var(stats_mean, stats_std, fig):
    """Multiple runs (random seeds) for a single variable (e.g. \vartheta)"""

    metrics_to_plot = []
    for metric_key in METRIC_ORDER:
        if metric_key in stats_mean and len(stats_mean[metric_key]):
            metrics_to_plot.append(metric_key)

    n_subplots = len(metrics_to_plot)
    n_cols = 4
    n_rows = ceil(n_subplots / n_cols)
    plot_count = count(1)

    for metric_key in metrics_to_plot:
        print('Plotting metric {}'.format(metric_key))
        ax = fig.add_subplot(n_rows, n_cols, next(plot_count))
        metric_mean = stats_mean[metric_key]
        color = DEFAULT_MEAN_COLOR
        ax.plot(metric_mean, color=color, label=None, lw=2)
        metric_std = 1 * stats_std[metric_key]
        lb, ub = metric_mean - metric_std, metric_mean + metric_std
        color = DEFAULT_STD_COLOR
        ax.plot(lb, color=color)
        ax.plot(ub, color=color)
        ax.fill_between(range(len(lb)), lb, ub, facecolor=color, alpha=0.5)
        ax.set_xlabel(X_LABEL_DEFAULT)
        ax.set_ylabel(PLOT_LABELS[metric_key])


def plot_multi_run_multi_var(stats_list, fig):
    # Multiple runs (random seeds) for each value a variable takes

    headers = [r'$\vartheta$', 'Runtime (s)', 'Est. error (%)',
               'Test error (%)']
    table = []
    for i in range(len(stats_list)):
        _, var_value, stats_value_mean, stats_value_std = stats_list[i]
        print('\n\nVar value = {}'.format(var_value))
        runtime = stats_value_mean['runtime']
        print('Runtime: {}'.format(runtime))

        est_err_mean = stats_value_mean['clf_error_mixed'][-1]
        est_err_std  = stats_value_std['clf_error_mixed'][-1]
        print('est_err: {} ({})'.format(est_err_mean, est_err_std))
        s1 = '{:5.2f} ({:4.2f})'.format(est_err_mean*100, est_err_std*100)

        test_err_mean = stats_value_mean.get('test_error_ilp', None)
        test_err_std = stats_value_std.get('test_error_ilp', None)
        print('test_err: {}'.format(test_err_mean, test_err_std))
        if test_err_mean is None:
            s2 = '-'
        else:
            s2 = '{:5.2f} ({:4.2f})'.format(test_err_mean*100, test_err_std*100)

        row = [var_value, runtime, s1, s2]
        table.append(row)

    print(tabulate(table, headers, tablefmt="latex"))

    metrics_to_plot = []
    _, _, stats_value_mean0, _ = stats_list[0]
    for metric_key in METRIC_ORDER:
        metric = stats_value_mean0.get(metric_key, None)
        if metric is not None and len(metric):
            metrics_to_plot.append(metric_key)

    n_subplots = len(metrics_to_plot)
    n_cols = 4
    n_rows = ceil(n_subplots / n_cols)
    plot_count = count(1)

    for metric_key in metrics_to_plot:
        print('Plotting metric {}'.format(metric_key))
        ax = fig.add_subplot(n_rows, n_cols, next(plot_count))

        for i in range(len(stats_list)):
            _, var_value, stats_value_mean, stats_value_std = stats_list[i]

            metric_mean = stats_value_mean[metric_key]
            color = DEFAULT_MEAN_COLOR
            ax.plot(metric_mean, color=color, label=None, lw=2)
            metric_std = 1 * stats_value_std[metric_key]
            lb, ub = metric_mean - metric_std, metric_mean + metric_std
            color = DEFAULT_STD_COLOR
            ax.plot(lb, color=color)
            ax.plot(ub, color=color)
            ax.fill_between(range(len(lb)), lb, ub, facecolor=color, alpha=0.3)
            ax.set_xlabel(X_LABEL_DEFAULT)
            ax.set_ylabel(PLOT_LABELS[metric_key])


def plot_standard(single_run, single_var, stats, config, title, path):

    figsize = (11, 5)
    fig = plt.figure(1, figsize=figsize, dpi=200)

    print()
    if single_run and single_var:  # default_run
        print('Plotting a single run with a single variable value')
        plot_single_run_single_var(stats[0], fig, config)
    elif single_run and not single_var:  # var_theta
        print('Plotting a single run for each variable value')
        plot_single_run_multi_var(stats, fig, config)
    elif single_var and not single_run:  # mean and std of default_run
        print('Plotting multiple runs for a single variable value')
        plot_multi_run_single_var(stats[0], stats[1], fig)
    elif not single_run and not single_var:
        print('Plotting multiple runs for multiple variable values')
        plot_multi_run_multi_var(stats, fig)
    print()

    # plt.legend(loc='upper right')
    dataset = config['dataset']['name']
    dataset = 'kitti' if dataset.startswith('kitti') else dataset
    dataset = dataset.replace('_', ' ')
    if title == r'Default run' or dataset == 'kitti':
        title = dataset.upper()
    else:
        title = r'\textbf{' + title + r'}' + ' (' + dataset.upper() + ')'

    fig.suptitle(title, fontsize='xx-large', fontweight='bold')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if path is not None:
        ori = 'landscape'
        if path.endswith('.stat'):
            path = path[:-5]

        try:
            fig.savefig(path + '.pdf', orientation=ori, transparent=False)
        except RuntimeError as e:
            print('Cannot save figure due to: \n{}'.format(e))


def plot_curves(stats, config, title='', path=None):
    plt.rc('text', usetex=True)  # need dvipng in Ubuntu
    plt.rc('font', family='serif')
    # matplotlib.rcParams['text.latex.unicode'] = True

    if type(stats) is tuple:
        single_var = True
        stats_mean, stats_std = stats
        single_run = stats_std is None
        stats_print = stats_mean
    elif type(stats) is list:
        single_var = False
        var_name, var_value, stats_mean0, stats_std0 = stats[0]
        print('Plotting multi var: {}'.format(var_name))
        single_run = stats_std0 is None
        stats_print = stats_mean0
    else:
        print('stats has type: {}'.format(type(stats).__name__))
        raise TypeError('stats is neither list nor tuple!')

    print('\nStatistics{}Size{}Type\n{}'.format(' '*25, ' '*8, '-'*52))
    for k in sorted(stats_print.keys()):
        v = stats_print[k]
        if v is not None:
            if hasattr(v, '__len__'):
                print('{:>32} {:>8} {:>9}'.format(k, len(v), type(v).__name__))
            else:
                print('{:>32} {:>8.2f} {:>9}'.format(k, v, type(v).__name__))

    if config is not None:
        print_config(config)

    plot_standard(single_run, single_var, stats, config, title, path)

    if single_var:
        if config['dataset']['name'] == 'mnist':
            test_pred_knn = stats_mean.get('test_pred_knn', None)
            if test_pred_knn is not None and len(test_pred_knn) > 0:
                plot_confusion_matrices(stats_mean, config)
