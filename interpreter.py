import pandas as pd
from typing import List


class Interpreter:

    def __init__(self, y_pred: List[str], y_true: List[str]):
        self.y_pred = y_pred
        self.y_true = y_true

    def calculate_accuracy(self) -> float:
        return sum(self.is_correct()) / len(self.y_true)

    def calculate_confusion_matrix(self) -> pd.DataFrame:
        return pd.crosstab(pd.Series(self.y_true, name='True'), pd.Series(self.y_pred, name='Prediction'))

    def calculate_accuracy_by_label(self) -> pd.DataFrame:
        df_is_correct = pd.DataFrame(dict(label=self.y_true, is_correct=self.is_correct()))
        accuracy_by_label = df_is_correct.groupby('label', as_index=False).mean()
        return accuracy_by_label

    def get_misclassified_samples(self) -> pd.Series:
        df_is_correct = pd.DataFrame(dict(label=self.y_true, is_correct=self.is_correct()))
        return df_is_correct[df_is_correct.is_correct == False].label

    def is_correct(self) -> List[bool]:
        return ([t == p for t, p in zip(self.y_true, self.y_pred)])
