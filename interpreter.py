from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Interpreter:

    def __init__(self,
                 images: list,
                 y_pred: np.ndarray,
                 y_true: np.ndarray,
                 probabilities: np.ndarray,
                 mapper: Dict[int, str]):

        self.images = images
        self.y_true = y_true
        self.y_pred = y_pred
        self.mapper = mapper
        self.probabilities = probabilities

        self.y_pred_labels = self.classes_to_labels(y_pred)
        self.y_true_labels = self.classes_to_labels(y_true)

    def calculate_accuracy(self) -> float:
        return sum(self.is_correct()) / len(self.y_true)

    def calculate_confusion_matrix(self) -> pd.DataFrame:
        return pd.crosstab(pd.Series(self.y_true_labels, name='True'),
                           pd.Series(self.y_pred_labels, name='Prediction'))

    def calculate_accuracy_by_label(self) -> pd.DataFrame:
        df_is_correct = pd.DataFrame(dict(label=self.y_true_labels, is_correct=self.is_correct()))
        accuracy_by_label = df_is_correct.groupby('label').mean()
        return accuracy_by_label

    def get_misclassified_samples(self) -> pd.DataFrame:
        summary_table = self.get_summary_table()
        return summary_table[summary_table.is_correct == False]

    def is_correct(self) -> np.ndarray:
        return self.y_pred == self.y_true

    def get_most_uncertain_samples(self, n_samples: int = None) -> pd.DataFrame:
        probabilities = self.get_propabilities_of_predicted_classes()
        indices = np.argsort(probabilities)
        if n_samples:
            indices = indices[:n_samples]
        data = [(i, probabilities[i]) for i in indices]
        return pd.DataFrame(data, columns=['sample', 'probability']).set_index('sample')

    def get_most_incorrect_samples(self, n_samples: int = None) -> pd.DataFrame:
        probabilities = self.get_propabilities_of_true_classes()
        indices = np.argsort(probabilities)
        if n_samples:
            indices = indices[:n_samples]
        data = [(i, probabilities[i]) for i in indices]
        return pd.DataFrame(data, columns=['sample', 'probability']).set_index('sample')

    def classes_to_labels(self, classes: np.ndarray) -> np.ndarray:
        return np.array([self.mapper[item] for item in classes])

    def get_propabilities_of_true_classes(self) -> np.ndarray:
        return np.array([p[i] for p, i in zip(self.probabilities, self.y_true)])

    def get_propabilities_of_predicted_classes(self) -> np.ndarray:
        return np.max(self.probabilities, axis=1)

    def get_summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            dict(true_label=self.y_true_labels,
                 predicted_label=self.y_pred_labels,
                 propability_true=self.get_propabilities_of_true_classes(),
                 propability_predicted=self.get_propabilities_of_predicted_classes(),
                 is_correct=self.is_correct(),
                 )
        )

    def plot_prediction(self, sample: int, n_classes: int = 5):
        propabilities = self.probabilities[sample, :]
        indices = np.argsort(propabilities)[-n_classes:]
        propabilities = propabilities[indices]
        labels = np.array(list(self.mapper.values()))[indices]
        image_array = np.array(self.images[sample])
        true_label = self.y_true_labels[sample]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.barh(labels, propabilities)
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.imshow(image_array)
        plt.title(f' True label: {true_label}')

    def plot_most_incorrect_samples(self, n_samples: int):
        samples = self.get_most_incorrect_samples(n_samples).index
        for sample in samples:
            self.plot_prediction(sample)
