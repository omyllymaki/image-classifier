from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from interpreters.base_interpreter import BaseInterpreter


class SingleLabelInterpreter(BaseInterpreter):
    def __init__(self,
                 images: list,
                 y_pred: list,
                 y_true: list,
                 probabilities: list,
                 mapper: Dict[int, str]):
        self.y_pred_flat = np.array(y_pred).flatten()
        self.y_true_flat = np.array(y_true).flatten()
        super().__init__(images, y_pred, y_true, probabilities, mapper)

    def calculate_accuracy(self) -> float:
        return self.calculate_is_correct_array().mean()

    def calculate_accuracy_by_label(self) -> pd.Series:
        series = pd.Series(self.calculate_is_correct_array(), index=self.y_true_flat)
        return series.groupby(series.index).mean().rename(self.mapper)

    def calculate_confusion_matrix(self) -> pd.DataFrame:
        labels = self.get_labels()
        result = confusion_matrix(self.y_true_flat, self.y_pred_flat)
        return pd.DataFrame(result, index=labels, columns=labels)

    def get_summary_table(self) -> pd.DataFrame:
        df_summary = super().get_summary_table()
        df_summary['confidence'] = np.sqrt(self.probabilities.shape[1]) * self.probabilities.std(axis=1, ddof=1)
        return df_summary

    def calculate_is_correct_array(self) -> np.ndarray:
        return self.y_pred_flat == self.y_true_flat