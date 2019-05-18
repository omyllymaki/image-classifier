import itertools
import logging
import time

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def test_hyperparameters(learner,
                         image_data,
                         hyperparameter_grid):
    df = pd.DataFrame(itertools.product(*hyperparameter_grid.values()), columns=hyperparameter_grid.keys())
    lowest_validation_losses, times = [], []
    for row in df.itertuples():
        test_learner = learner
        parameters = row._asdict()
        index = parameters.pop('Index')
        logger.info(f'{index+1}/{df.shape[0]}')
        time_started = time.time()
        losses, losses_valid = test_learner.fit_model(image_data, **parameters)
        time_ended = time.time()
        lowest_validation_losses.append(np.min(losses_valid))
        times.append(time_ended - time_started)

    df['lowest_validation_loss'] = lowest_validation_losses
    df['time'] = times

    return df
