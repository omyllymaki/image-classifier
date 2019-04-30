from typing import Dict

import numpy as np
import pandas as pd

from interpreters.base_interpreter import BaseInterpreter


class MultiLabelInterpreter(BaseInterpreter):
    def __init__(self,
                 images: list,
                 y_pred: list,
                 y_true: list,
                 probabilities: list,
                 mapper: Dict[int, str]):
        super().__init__(images, y_pred, y_true, probabilities, mapper)

    def calculate_accuracy(self) -> float:
        return self.is_correct_array.mean()

    def calculate_accuracy_by_label(self) -> pd.Series:
        accuracy_by_label = self.is_correct_array.mean(axis=0)
        return pd.Series(accuracy_by_label).rename(self.mapper)

    def calculate_confusion_matrix(self):
        print('Confusion matrix is not implemented for multilabel classification')
        return None

    def get_summary_table(self):
        df_summary = super().get_summary_table()
        df_summary['confidence'] = abs(self.probabilities - 0.5).mean(axis=1) / 0.5
        return df_summary

    def calculate_is_correct_array(self) -> np.ndarray:
        return self.y_pred_one_hot_encoded == self.y_true_one_hot_encoded