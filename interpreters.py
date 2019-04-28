from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from plotting import show_image

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

        one_hot_encoder = MultiLabelBinarizer()
        self.y_pred_one_hot_encoded = one_hot_encoder.fit_transform(y_pred)
        self.y_true_one_hot_encoded = one_hot_encoder.fit_transform(y_true)
        self.is_correct_array = self.calculate_is_correct_array()

    def calculate_accuracy(self) -> float:
        return self.is_correct_array.mean()

    def calculate_accuracy_by_label(self) -> pd.Series:
        accuracy_by_label = self.is_correct_array.mean(axis=0)
        return pd.Series(accuracy_by_label).rename(self.mapper)

    def calculate_is_correct_array(self) -> np.ndarray:
        return self.y_pred_one_hot_encoded == self.y_true_one_hot_encoded

    def get_summary_table(self) -> pd.DataFrame:
        probability_differences = abs(self.probabilities - self.y_true_one_hot_encoded)
        mean_difference_per_sample = probability_differences.mean(axis=1)
        df_pred = pd.DataFrame(self.probabilities).rename(self.mapper, axis='columns').add_suffix('_prediction')
        df_true = pd.DataFrame(self.y_true_one_hot_encoded).rename(self.mapper, axis='columns').add_suffix('_true')
        df_mean_difference = pd.DataFrame(dict(prediction_error=mean_difference_per_sample))
        df_summary = df_mean_difference.join(df_pred).join(df_true)
        df_summary['confidence'] = abs(df_pred - 0.5).mean(axis=1) / 0.5
        df_summary['accuracy'] = self.is_correct_array.mean(axis=1)
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
        labels = np.array(list(self.mapper.values()))[indices]
        image = self.images[sample]
        true_label = [self.mapper[item] for item in self.y_true[sample]]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.barh(labels, probabilities)
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

    def calculate_confusion_matrix(self):
        print('Not implemented')
        return None


# class BaseInterpreter:
#     def __init__(self,
#                  images: list,
#                  y_pred: np.ndarray,
#                  y_true: np.ndarray,
#                  probabilities: np.ndarray,
#                  mapper: Dict[int, str]):
#         self.images = images
#         self.y_true = y_true
#         self.y_pred = y_pred
#         self.mapper = mapper
#         self.probabilities = probabilities
#
#     def calculate_accuracy(self) -> float:
#         raise NotImplementedError
#
#     def calculate_accuracy_by_label(self) -> pd.Series:
#         raise NotImplementedError
#
#     def get_summary_table(self) -> pd.DataFrame:
#         raise NotImplementedError
#
#     def get_most_incorrect_samples(self, n_samples: int) -> pd.DataFrame:
#         raise NotImplementedError
#
#     def get_most_uncertain_samples(self, n_samples: int) -> pd.DataFrame:
#         raise NotImplementedError
#
#     def plot_prediction(self, sample: int, n_classes: int = 5) -> None:
#         raise NotImplementedError
#
#     def plot_most_incorrect_samples(self, n_samples: int) -> None:
#         raise NotImplementedError
#
#     def plot_most_uncertain_samples(self, n_samples: int):
#         raise NotImplementedError
#
#     def calculate_confusion_matrix(self) -> pd.DataFrame:
#         raise NotImplementedError
#
#
# class SinglelabelInterpreter(BaseInterpreter):
#
#     def __init__(self,
#                  images: list,
#                  y_pred: np.ndarray,
#                  y_true: np.ndarray,
#                  probabilities: np.ndarray,
#                  mapper: Dict[int, str]):
#
#         super().__init__(images, y_pred, y_true, probabilities, mapper)
#         self.y_pred_labels = self.classes_to_labels(y_pred)
#         self.y_true_labels = self.classes_to_labels(y_true)
#
#     def calculate_accuracy(self) -> float:
#         return sum(self.is_correct()) / len(self.y_true)
#
#     def calculate_confusion_matrix(self) -> pd.DataFrame:
#         return pd.crosstab(pd.Series(self.y_true_labels, name='True'),
#                            pd.Series(self.y_pred_labels, name='Prediction'))
#
#     def calculate_accuracy_by_label(self) -> pd.DataFrame:
#         df_is_correct = pd.DataFrame(dict(label=self.y_true_labels, is_correct=self.is_correct()))
#         accuracy_by_label = df_is_correct.groupby('label').mean()
#         return accuracy_by_label
#
#     def get_misclassified_samples(self) -> pd.DataFrame:
#         summary_table = self.get_summary_table()
#         return summary_table[summary_table.is_correct == False]
#
#     def is_correct(self) -> np.ndarray:
#         return self.y_pred == self.y_true
#
#     def get_most_uncertain_samples(self, n_samples: int = None) -> pd.DataFrame:
#         probabilities = self.get_propabilities_of_predicted_classes()
#         indices = np.argsort(probabilities)
#         if n_samples:
#             indices = indices[:n_samples]
#         data = [(i, probabilities[i]) for i in indices]
#         return pd.DataFrame(data, columns=['sample', 'probability']).set_index('sample')
#
#     def get_most_incorrect_samples(self, n_samples: int = None) -> pd.DataFrame:
#         probabilities = self.get_propabilities_of_true_classes()
#         indices = np.argsort(probabilities)
#         if n_samples:
#             indices = indices[:n_samples]
#         data = [(i, probabilities[i]) for i in indices]
#         return pd.DataFrame(data, columns=['sample', 'probability']).set_index('sample')
#
#     def classes_to_labels(self, classes: np.ndarray) -> np.ndarray:
#         return np.array([self.mapper[item] for item in classes])
#
#     def get_propabilities_of_true_classes(self) -> np.ndarray:
#         return np.array([p[i] for p, i in zip(self.probabilities, self.y_true)])
#
#     def get_propabilities_of_predicted_classes(self) -> np.ndarray:
#         return np.max(self.probabilities, axis=1)
#
#     def get_summary_table(self) -> pd.DataFrame:
#         return pd.DataFrame(
#             dict(true_label=self.y_true_labels,
#                  predicted_label=self.y_pred_labels,
#                  propability_true=self.get_propabilities_of_true_classes(),
#                  propability_predicted=self.get_propabilities_of_predicted_classes(),
#                  is_correct=self.is_correct(),
#                  )
#         )
#
#     def plot_prediction(self, sample: int, n_classes: int = 5):
#         propabilities = self.probabilities[sample, :]
#         indices = np.argsort(propabilities)[-n_classes:]
#         propabilities = propabilities[indices]
#         labels = np.array(list(self.mapper.values()))[indices]
#         image = self.images[sample]
#         true_label = self.y_true_labels[sample]
#
#         plt.figure()
#         plt.subplot(1, 2, 1)
#         plt.barh(labels, propabilities)
#         plt.grid()
#         plt.subplot(1, 2, 2)
#         show_image(image, f' True label: {true_label}')
#
#     def plot_most_incorrect_samples(self, n_samples: int):
#         samples = self.get_most_incorrect_samples(n_samples).index
#         for sample in samples:
#             self.plot_prediction(sample)
#
#     def plot_most_uncertain_samples(self, n_samples: int):
#         samples = self.get_most_uncertain_samples(n_samples).index
#         for sample in samples:
#             self.plot_prediction(sample)
#
#
# def get_interpreter(images, y_pred, y_true, probabilities, mapper, is_multilabel: bool):
#     if is_multilabel:
#         interpreter = MultilabelInterpreter(images, y_pred, y_true, probabilities, mapper)
#     else:
#         interpreter = MultilabelInterpreter(images, y_pred, y_true, probabilities, mapper)
#     return interpreter
