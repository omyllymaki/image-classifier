import itertools
import logging
import time
import random
from typing import Dict

import pandas as pd
import numpy as np

from image_data import ImageData
from learners.base_learner import BaseLearner

logger = logging.getLogger(__name__)


class ModelOptimizer:

    def __init__(self, learner: BaseLearner,
                 data: ImageData,
                 parameter_options: Dict[str, list]):
        self.learner = learner
        self.data = data
        self.parameter_options = parameter_options
        self.learner_candidates = None
        self.results = None

    def run_grid_search(self) -> pd.DataFrame:
        df_parameters = self._get_parameter_table_for_grid_search()
        return self._run_search(df_parameters)

    def run_random_search(self, n_iterations: int = 10) -> pd.DataFrame:
        df_parameters = self._get_parameter_table_for_random_search(n_iterations)
        return self._run_search(df_parameters)

    def get_best_learner(self) -> BaseLearner:
        index = self.results.iloc[0].name
        return self.learner_candidates[index]

    def get_best_parameters(self) -> dict:
        best_results = self.results.iloc[0]
        return best_results[self.parameter_options.keys()].to_dict()

    def _run_search(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _get_parameter_table_for_grid_search(self) -> pd.DataFrame:
        return pd.DataFrame(itertools.product(*self.parameter_options.values()),
                            columns=self.parameter_options.keys())

    def _get_parameter_table_for_random_search(self, n_iterations: int) -> pd.DataFrame:
        parameters = []
        for _ in range(n_iterations):
            parameter_set = {}
            for parameter_name, values in self.parameter_options.items():
                parameter_set[parameter_name] = random.choice(values)
            parameters.append(parameter_set)
        return pd.DataFrame(parameters)
