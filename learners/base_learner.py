import logging
from abc import abstractmethod
from typing import Tuple, Callable, List

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class BaseLearner:

    def __init__(self,
                 model,
                 loss_function: Callable,
                 optimizer_function: Callable = torch.optim.Adam,
                 use_gpu: bool = False):
        self.model = model
        self.loss_function = loss_function()
        self.optimizer_function = optimizer_function
        self.use_gpu = use_gpu
        self.device = self.get_device()
        self.epoch = None
        self.batch_index = None
        self.models = None
        self.best_model = None

    def get_device(self):
        if self.use_gpu and torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def fit_model(self,
                  data,
                  image_transforms_training,
                  image_transforms_validation,
                  batch_size=1,
                  epochs=1,
                  learning_rate: float = 0.001,
                  weight_decay: float = 0.01,
                  early_stop_option=True):

        self.model = self.model.to(self.device)
        self.optimizer = self.optimizer_function(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.set_traininig_mode()
        self.class_to_label_mapping = data.class_to_label_mapping
        self.early_stop_option = early_stop_option
        self.epochs = epochs
        self.image_data = data
        self.one_hot_encoder = MultiLabelBinarizer(classes=list(self.class_to_label_mapping.keys()))

        x_valid, y_valid = self.prepare_validation_data(data, image_transforms_validation)

        self.losses, self.validation_losses, self.models = [], [], []
        for self.epoch in range(1, self.epochs + 1):
            batches = data.make_batches('training', batch_size=batch_size)
            for self.batch_index, batch in enumerate(batches):
                x_batch, y_batch = self.prepare_training_data(batch, data, image_transforms_training)

                self.calculate_training_loss(x_batch, y_batch)
                self.update_weights()
                self.log_batch()

            self.calculate_validation_loss(x_valid, y_valid)
            self.models.append(self.model)
            self.log_epoch()

            if self.is_stop_criteria_filled():
                logger.info('Early stop criterion filled; fitting completed!')
                break

        self.best_model = self.get_best_model()
        return self.losses, self.validation_losses

    def get_best_model(self):
        lowest_validation_loss_index = int(np.argmin(self.validation_losses))
        lowest_validation_loss = self.validation_losses[lowest_validation_loss_index]
        logger.info(f'Lowest validation loss: epoch {lowest_validation_loss_index + 1}; loss {lowest_validation_loss}')
        return self.models[lowest_validation_loss_index]

    def calculate_training_loss(self, x_batch, y_batch):
        y_predicted = self.model(x_batch)
        self.loss = self.loss_function(y_predicted, y_batch)
        self.losses.append(self.loss.item())

    def prepare_training_data(self, batch, data, image_transforms_training):
        x_batch, y_batch = data.get_batch_data(batch)
        x_batch = self.apply_transforms_to_images(x_batch, image_transforms_training)
        y_batch = self.classes_to_target_tensor(y_batch)
        return x_batch.to(self.device), y_batch.to(self.device)

    def prepare_validation_data(self, data, image_transforms_validation):
        x_valid = data.get_images('validation')
        y_valid = data.get_classes('validation')
        x_valid = self.apply_transforms_to_images(x_valid, image_transforms_validation)
        y_valid = self.classes_to_target_tensor(y_valid)
        return x_valid.to(self.device), y_valid.to(self.device)

    def log_epoch(self):
        logger.info(
            f'''
            Epoch: {self.epoch}/{self.epochs}
            Validation loss: {self.loss_valid.item()}''')

    def log_batch(self):
        logger.debug(
            f'''
            Epoch: {self.epoch}/{self.epochs}
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

    def is_stop_criteria_filled(self, n_epochs: int = 3):
        if not self.early_stop_option:
            return False
        if self.epoch < n_epochs:
            return False
        last_losses = self.validation_losses[-n_epochs:]
        slope = np.polyfit(range(len(last_losses)), last_losses, 1)[0]
        return slope > 0

    def set_traininig_mode(self):
        self.model.train()

    def set_evaluation_mode(self):
        self.model.eval()  # E.g. disables dropout

    def predict(self, images: list, transforms, **kwargs) -> Tuple[list, list]:
        self.set_evaluation_mode()
        predicted_classes, probabilities = [], []
        for image in images:
            image = transforms(image)
            image = image.unsqueeze(0).to(self.device)
            prediction = self.best_model(image).cpu()
            prob = torch.exp(prediction).detach().numpy()[0]
            predicted_class = self.get_predicted_classes(prob, **kwargs)
            predicted_classes.append(predicted_class)
            probabilities.append(prob)
        return predicted_classes, probabilities

    @staticmethod
    def apply_transforms_to_images(images, transforms):
        return torch.stack([transforms(image) for image in images])

    @abstractmethod
    def classes_to_target_tensor(self, classes_list: List[int]) -> torch.Tensor:
        """
        This needs to be implemented by inheritor.

        Single-label: should return 1-dimensional torch LongTensor where elements are classes.
        Multi-label: should return 2-dimensional torch Tensor where rows are one-hot-encoded classes.
        """
        raise NotImplementedError

    @abstractmethod
    def get_predicted_classes(self, probabilities, **kwargs) -> List[int]:
        """
        This needs to be implemented by inheritor.

        For single-label and multi-label, should return list of predicted classes (integers).
        """
        raise NotImplementedError
