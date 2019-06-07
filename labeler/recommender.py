from typing import List

import numpy as np

from constants import BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, USE_EARLY_STOP
from image_data import ImageData
from image_transforms import TransformsTraining, TransformsTest


class Recommender:

    def __init__(self, learner):
        self.p_train = 0.75
        self.p_valid = 0.25
        self.min_valid_loss = None
        self.learner = learner

    def query(self, images: list):
        confidences = self._evaluate_unlabeled(images)
        sorted_indices = self._get_most_uncertain_samples(confidences)
        return sorted_indices

    def train_model(self, data: List[dict]):
        image_data = ImageData(data, self.p_train, self.p_valid, 0.0)
        losses, losses_valid = self.learner.fit_model(image_data,
                                                      image_transforms_training=TransformsTraining,
                                                      image_transforms_validation=TransformsTest,
                                                      batch_size=BATCH_SIZE,
                                                      epochs=EPOCHS,
                                                      learning_rate=LEARNING_RATE,
                                                      weight_decay=WEIGHT_DECAY,
                                                      early_stop_option=USE_EARLY_STOP)
        self.min_valid_loss = np.min(losses_valid)

    def _evaluate_unlabeled(self, images: list):
        y_pred, probabilities = self.learner.predict(images, TransformsTest)
        probabilities = np.array(probabilities)
        confidences = np.sqrt(probabilities.shape[1]) * probabilities.std(axis=1, ddof=1)
        return confidences

    def _get_most_uncertain_samples(self, confidences: List[float]):
        return np.argsort(confidences)
