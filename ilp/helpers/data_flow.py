import numpy as np
from sklearn.utils.validation import check_random_state


def check_min_samples_per_class(y, min_samples=2):
    classes, class_sizes = np.unique(y, return_counts=True)
    min_class_size = class_sizes.min()
    print('Class sizes: {}'.format(class_sizes))
    if min_class_size < min_samples:
        print('Classes: {}'.format(np.unique(y)))
        raise ValueError('Minimum class size < 2.')

    return classes, class_sizes


def split_burn_in_rest(y, n_labeled_per_class, ratio_labeled, shuffle=False, seed=None):
    """

    Parameters
    ----------
    y : array, shape (n_samples,)
        The true data labels.

    n_labeled_per_class : int
        Number of labeled samples per class to include in the burn-in set.

    ratio_labeled : float 
        Ratio of labeled samples to include within the burn-in set.

    shuffle : bool
        Whether to shuffle indices within classes before adding to burn-in set.

    seed : int, np.random.RandomState or None
        For reproducibility.

    Returns
    -------
    ind_burn_in : list of length n_burn_in
        Indices of the samples in the burn-in set.

    mask_labeled_burn_in : array, shape (n_burn_in,)
        Mask indicating whether the labels in the burn-in set are observed.

    """
    classes, class_sizes = check_min_samples_per_class(y, n_labeled_per_class)

    rng = check_random_state(seed)
    ind_burn_in = []
    set_ind_burn_in_labeled = set()

    for class_ in classes:
        ind_class, = np.where(y == class_)
        n_class = len(ind_class)
        n_unlabeled_class = int(
            n_labeled_per_class * (1 / ratio_labeled - 1)) #+ 1
        n_unlabeled_class = min(n_unlabeled_class,
                                n_class - n_labeled_per_class)
        n_burn_in_class = n_labeled_per_class + n_unlabeled_class

        if shuffle:
            ind_samples = rng.choice(n_class, n_burn_in_class, replace=False)
            ind_burn_in_class = ind_class[ind_samples]
            ind_burn_in.extend(ind_burn_in_class)
            ind_samples = rng.choice(n_burn_in_class, n_labeled_per_class,
                                     replace=False)
            ind_burn_in_class_labeled = ind_burn_in_class[ind_samples]
        else:
            ind_burn_in_class = ind_class[:n_burn_in_class]
            ind_burn_in.extend(ind_burn_in_class)
            ind_burn_in_class_labeled = ind_burn_in_class[:n_labeled_per_class]

        set_ind_burn_in_labeled.update(ind_burn_in_class_labeled)

    mask_labeled_burn_in = [i in set_ind_burn_in_labeled for i in ind_burn_in]
    mask_labeled_burn_in = np.asarray(mask_labeled_burn_in)

    y_burn_in = y[ind_burn_in]
    y_burn_in_labeled = y_burn_in[mask_labeled_burn_in]
    y_burn_in_unlabeled = y_burn_in[~mask_labeled_burn_in]

    _, class_sizes_labeled = np.unique(y_burn_in_labeled, return_counts=True)
    _, class_sizes_unlabeled = np.unique(y_burn_in_unlabeled,
                                         return_counts=True)

    if len(y_burn_in_labeled) == 0 and len(y_burn_in_unlabeled) == 0:
        class_sizes_burnin = np.zeros(len(classes))
    elif len(y_burn_in_labeled) == 0:
        class_sizes_burnin = class_sizes_unlabeled
    elif len(y_burn_in_unlabeled) == 0:
        class_sizes_burnin = class_sizes_labeled
    else:
        class_sizes_burnin = class_sizes_labeled + class_sizes_unlabeled

    print('\n\n')
    print('Burn-in   labeled class sizes: {} , sum = {}'.format(
        class_sizes_labeled, sum(class_sizes_labeled)))
    print('Burn-in unlabeled class sizes: {}, sum = {}'.format(
        class_sizes_unlabeled, sum(class_sizes_unlabeled)))
    print('Burn-in     total class sizes: {}, sum = {}'.format(
        class_sizes_burnin, sum(class_sizes_burnin)))
    print('\nRest total size: {}'.format(len(y) - len(y_burn_in)))

    return ind_burn_in, mask_labeled_burn_in


def split_labels_rest(y_rest, ratio_labeled, batch_size, shuffle=False,
                      seed=None):
    """

    Parameters
    ----------
    y_rest : array with shape (n_rest,)
        Remaining data labels after burn-in.

    ratio_labeled : float
        Ratio of observed labels in remaining data.

    batch_size : int
        Number of points for which the ratio_labeled must be satisfied.

    shuffle : bool
        Whether to shuffle indices within classes before adding to burn-in set.

    seed : int, np.random.RandomState or None
        For reproducibility.

    Returns
    -------
    mask_labeled_rest : array, shape (n_rest,)
        Mask indicating whether the labels in the rest set are observed.

    """

    classes = np.unique(y_rest)

    rng = check_random_state(seed)

    set_ind_rest_labeled = set()

    for class_ in classes:
        ind_class, = np.where(y_rest == class_)
        n_class = len(ind_class)
        n_labeled_class = int(n_class * ratio_labeled)

        if shuffle:
            ind_samples = rng.choice(n_class, n_labeled_class, replace=False)
            is_labeled_rest_class = ind_class[ind_samples]
        else:
            is_labeled_rest_class = ind_class[:n_labeled_class]

        set_ind_rest_labeled.update(is_labeled_rest_class)

    mask_labeled_rest = [i in set_ind_rest_labeled for i in range(len(y_rest))]
    mask_labeled_rest = np.asarray(mask_labeled_rest)

    y_rest_labeled = y_rest[mask_labeled_rest]
    y_rest_unlabeled = y_rest[~mask_labeled_rest]

    _, class_sizes_labeled = np.unique(y_rest_labeled, return_counts=True)
    _, class_sizes_unlabeled = np.unique(y_rest_unlabeled, return_counts=True)

    if len(y_rest_labeled) == 0 and len(y_rest_unlabeled) == 0:
        class_sizes_rest = np.zeros(len(classes))
    elif len(y_rest_labeled) == 0:
        class_sizes_rest = class_sizes_unlabeled
    elif len(y_rest_unlabeled) == 0:
        class_sizes_rest = class_sizes_labeled
    else:
        class_sizes_rest = class_sizes_labeled + class_sizes_unlabeled

    print('\n\n')
    print('Rest   labeled class sizes: {}, sum = {}'.format(
        class_sizes_labeled, sum(class_sizes_labeled)))
    print('Rest unlabeled class sizes: {}, sum = {}'.format(
        class_sizes_unlabeled, sum(class_sizes_unlabeled)))
    print('Rest     total class sizes: {}, sum = {}'.format(
        class_sizes_rest, sum(class_sizes_rest)))
    print('\n\n')

    return mask_labeled_rest


def gen_semilabeled_data(inputs, targets, flags):
    """
    Generates a sequence of all inputs and targets, prepended with an id
    of the sample. A boolean value indicating if the label is observed by
    the algorithm or not is also generated.
    """

    assert len(inputs) == len(targets) == len(flags)

    indices = range(len(inputs))

    for i, j, k, l in zip(indices, inputs, targets, flags):
        yield i, j, k, l


def gen_data_stream(inputs, targets, shuffle=False, seed=None):
    """
    Generates a sequence of all inputs and targets, optionally shuffled.
    """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        random_state = check_random_state(seed)
        random_state.shuffle(indices)
        for i in indices:
            yield inputs[i], targets[i]
    else:
        for i in range(len(inputs)):
            yield inputs[i], targets[i]
