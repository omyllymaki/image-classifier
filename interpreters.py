from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from plotting import show_image


class BaseInterpreter:
    def __init__(self,
                 images: list,
                 y_pred: list,
                 y_true: list,
                 probabilities: list,
                 mapper: Dict[int, str]):
        self.images = images
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.mapper = mapper
        self.probabilities = np.array(probabilities)

        one_hot_encoder = MultiLabelBinarizer()
        self.y_pred_one_hot_encoded = one_hot_encoder.fit_transform(y_pred)
        self.y_true_one_hot_encoded = one_hot_encoder.fit_transform(y_true)
        self.is_correct_array = self.calculate_is_correct_array()

    def get_summary_table(self) -> pd.DataFrame:
        probability_differences = abs(self.probabilities - self.y_true_one_hot_encoded)
        mean_difference_per_sample = probability_differences.mean(axis=1)
        df_pred = pd.DataFrame(self.probabilities).rename(self.mapper, axis='columns').add_suffix('_prediction')
        df_true = pd.DataFrame(self.y_true_one_hot_encoded).rename(self.mapper, axis='columns').add_suffix('_true')
        df_mean_difference = pd.DataFrame(dict(prediction_error=mean_difference_per_sample))
        df_summary = df_mean_difference.join(df_pred).join(df_true)
        return df_summary

    def get_most_incorrect_samples(self, n_samples: int) -> pd.DataFrame:
        df_summary = self.get_summary_table()
        df_summary_sorted = df_summary.sort_values('prediction_error', ascending=False)
        return df_summary_sorted.iloc[:n_samples]

    def get_most_uncertain_samples(self, n_samples: int) -> pd.DataFrame:
        df_summary = self.get_summary_table()
        df_summary_sorted = df_summary.sort_values('confidence', ascending=True)
        return df_summary_sorted.iloc[:n_samples]

    def plot_prediction(self, sample: int, n_classes: int = 5):
        probabilities = self.probabilities[sample]
        indices = np.argsort(probabilities)[-n_classes:]
        probabilities = probabilities[indices]
        labels = self.get_labels()[indices]
        image = self.images[sample]
        true_label = [self.mapper[item] for item in self.y_true[sample]]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.barh(labels, probabilities)
        plt.xlim(0, 1)
        plt.grid()
        plt.subplot(1, 2, 2)
        show_image(image, f' True label: {true_label}')

    def plot_most_incorrect_samples(self, n_samples: int):
        samples = self.get_most_incorrect_samples(n_samples).index
        for sample in samples:
            self.plot_prediction(sample)

    def plot_most_uncertain_samples(self, n_samples: int):
        samples = self.get_most_uncertain_samples(n_samples).index
        for sample in samples:
            self.plot_prediction(sample)

    def get_labels(self) -> np.ndarray:
        return np.array(list(self.mapper.values()))

    def calculate_confusion_matrix(self) -> pd.DataFrame:
        raise NotImplementedError

    def calculate_accuracy(self) -> float:
        raise NotImplementedError

    def calculate_accuracy_by_label(self) -> pd.Series:
        raise NotImplementedError

    def calculate_is_correct_array(self) -> np.ndarray:
        raise NotImplementedError


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
