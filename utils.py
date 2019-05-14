from math import floor
from random import shuffle
import numpy as np


def get_indices(indices, p_training, p_validation, p_test, randomize=True):
    if randomize:
        shuffle(indices)

    n_train = floor(p_training * len(indices))
    n_validation = floor(p_validation * len(indices))
    n_test = floor(p_test * len(indices))

    indices_train = indices[:n_train]
    indices_validation = indices[n_train:n_train + n_validation]
    indices_test = indices[n_train + n_validation:n_train + n_validation + n_test]

    return indices_train, indices_validation, indices_test


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)
