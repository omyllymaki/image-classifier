import random

import numpy as np
import torch


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def set_random_seeds(random_seed):
    random.seed(random_seed)
    _ = torch.manual_seed(random_seed)
    np.random.seed(random_seed)
