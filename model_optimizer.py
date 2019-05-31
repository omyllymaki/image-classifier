import itertools
import logging
import time
import random

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelOptimizer:

    def __init__(self, learner, data, parameter_options):
        self.learner = learner
        self.data = data
        self.parameter_options = parameter_options
        self.learner_candidates = None
        self.results = None

    def run_grid_search(self):
        df_parameters = self._get_parameter_table_by_grid_search()
        return self._run_search(df_parameters)

    def run_random_search(self, n_iterations=10):
        df_parameters = self._get_parameter_table_by_random_search(n_iterations)
        return self._run_search(df_parameters)

    def get_best_learner(self):
        index = self.results.iloc[0].name
        return self.learner_candidates[index]

    def _run_search(self, df):
        lowest_validation_losses, times = [], []
        self.learner_candidates = []
        for row in df.itertuples():
            fresh_learner = self.learner
            parameters = row._asdict()
            index = parameters.pop('Index')
            logger.info(f'{index+1}/{df.shape[0]}')
            time_started = time.time()
            losses, losses_valid = fresh_learner.fit_model(self.data, **parameters)
            time_ended = time.time()
            lowest_validation_losses.append(np.min(losses_valid))
            times.append(time_ended - time_started)
            self.learner_candidates.append(fresh_learner)
        df['lowest_validation_loss'] = lowest_validation_losses
        df['time'] = times
        df = df.sort_values(by='lowest_validation_loss')
        self.results = df
        return df

    def _get_parameter_table_by_grid_search(self):
        return pd.DataFrame(itertools.product(*self.parameter_options.values()),
                            columns=self.parameter_options.keys())

    def _get_parameter_table_by_random_search(self, n_iterations):
        parameters = []
        for _ in range(n_iterations):
            parameter_set = {}
            for parameter_name, values in self.parameter_options.items():
                parameter_set[parameter_name] = random.choice(values)
            parameters.append(parameter_set)
        return pd.DataFrame(parameters)
