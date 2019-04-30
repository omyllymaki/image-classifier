from abc import abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


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
        self.show_image(image, f' True label: {true_label}')

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

    @staticmethod
    def show_image(image, title=None):
        image = plt.imshow(np.asarray(image))
        plt.axis('off')
        if title:
            plt.title(title)
        return image

    @abstractmethod
    def calculate_confusion_matrix(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def calculate_accuracy(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_accuracy_by_label(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def calculate_is_correct_array(self) -> np.ndarray:
        raise NotImplementedError
