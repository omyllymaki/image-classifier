import logging
from typing import Tuple, Callable

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from image_data import ImageData

logger = logging.getLogger(__name__)


class Learner:

    def __init__(self,
                 model,
                 loss_function: Callable = torch.nn.BCEWithLogitsLoss,
                 optimizer_function: Callable = torch.optim.Adam):
        self.model = model
        self.loss_function = loss_function()
        self.optimizer_function = optimizer_function
        self.epoch = None
        self.batch_index = None

    def fit_model(self,
                  data: ImageData,
                  image_transforms_training,
                  image_transforms_validation,
                  batch_size=1,
                  epochs=1,
                  learning_rate: float = 0.001,
                  weight_decay: float = 0.01,
                  early_stop_option=True):

        self.optimizer = self.optimizer_function(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.set_traininig_mode()
        self.class_to_label_mapping = data.class_to_label_mapping
        self.early_stop_option = early_stop_option
        self.epochs = epochs
        self.image_data = data
        self.one_hot_encoder = MultiLabelBinarizer(classes=list(self.class_to_label_mapping.keys()))

        x_valid, y_valid = self.prepare_validation_data(data, image_transforms_validation)

        self.losses, self.validation_losses = [], []
        for self.epoch in range(self.epochs):
            batches = data.make_batches('training', batch_size=batch_size)
            for self.batch_index, batch in enumerate(batches):
                x_batch, y_batch = self.prepare_training_data(batch, data, image_transforms_training)

                self.calculate_training_loss(x_batch, y_batch)
                self.update_weights()
                self.log_batch()

            self.calculate_validation_loss(x_valid, y_valid)
            self.log_epoch()

            if self.is_stop_criteria_filled():
                logger.info('Early stop criterion filled; fitting completed!')
                return self.losses, self.validation_losses

        return self.losses, self.validation_losses

    def calculate_training_loss(self, x_batch, y_batch):
        y_predicted = self.model(x_batch)
        self.loss = self.loss_function(y_predicted, y_batch)
        self.losses.append(self.loss.item())

    def prepare_training_data(self, batch, data, image_transforms_training):
        x_batch, y_batch = data.get_batch_data(batch)
        x_batch = self.apply_transforms_to_images(x_batch, image_transforms_training)
        y_batch = self.integers_to_tensor(y_batch)
        return x_batch, y_batch

    def prepare_validation_data(self, data, image_transforms_validation):
        x_valid = data.get_images('validation')
        y_valid = data.get_classes('validation')
        x_valid = self.apply_transforms_to_images(x_valid, image_transforms_validation)
        y_valid = self.integers_to_tensor(y_valid)
        return x_valid, y_valid

    def log_epoch(self):
        logger.info(
            f'''
            Epoch: {self.epoch + 1}/{self.epochs}
            Validation loss: {self.loss_valid.item()}''')

    def log_batch(self):
        logger.debug(
            f'''
            Epoch: {self.epoch + 1}/{self.epochs}
            Batch: {self.batch_index}
            Training loss: {self.loss.item()}''')

    def calculate_validation_loss(self, x_valid, y_valid):
        self.set_evaluation_mode()
        with torch.no_grad():
            y_predicted_valid = self.model(x_valid)
            loss_valid = self.loss_function(y_predicted_valid, y_valid)
        self.loss_valid = loss_valid
        self.validation_losses.append(self.loss_valid.item())
        self.set_traininig_mode()

    def update_weights(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def is_stop_criteria_filled(self):
        if not self.early_stop_option:
            return False
        if self.epoch < 1:
            return False
        return self.validation_losses[-1] > self.validation_losses[-2]

    def predict(self, images: list, transforms, threshold: float = 0.5) -> Tuple[list, list]:
        self.set_evaluation_mode()
        predicted_classes, probabilities = [], []
        for image in images:
            image = transforms(image)
            image = image.unsqueeze(0)
            prediction = self.model(image)
            prob = torch.exp(prediction).detach().numpy()[0]
            predicted_class = np.where(prob > threshold)
            predicted_classes.append(predicted_class[0].tolist())
            probabilities.append(prob)
        return predicted_classes, probabilities

    def set_traininig_mode(self):
        self.model.train()

    def set_evaluation_mode(self):
        # E.g. disables dropout
        self.model.eval()

    @staticmethod
    def apply_transforms_to_images(images, transforms):
        return torch.stack([transforms(image) for image in images])

    def integers_to_tensor(self, data):
        data_one_hot_encoded = self.one_hot_encoder.fit_transform(data)
        return torch.Tensor(data_one_hot_encoded)
