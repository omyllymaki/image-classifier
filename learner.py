import logging

import torch

from image_data import ImageData

logger = logging.getLogger(__name__)


class Learner:

    def __init__(self, model, loss_function=None, optimizer=None):
        self.model = model
        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = torch.nn.NLLLoss()
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters())

    def fit_model(self,
                  data: ImageData,
                  batch_size=1,
                  epochs=1,
                  image_transforms=None,
                  early_stop_option=True):

        self.epochs = epochs
        x_valid = data.get_images('validation')
        y_valid = data.get_classes('validation')
        if image_transforms:
            image_transforms_validation = image_transforms.get('validation')
            image_transforms_training = image_transforms.get('validation')

        if image_transforms_validation:
            x_valid = self.apply_transforms_to_images(x_valid, image_transforms_validation)
        x_valid = torch.stack(x_valid)
        y_valid = torch.Tensor(y_valid).long()

        losses, validation_losses = [], []
        for self.epoch in range(self.epochs):
            batches = data.make_batches('training', batch_size=batch_size)

            for self.batch_index, batch in enumerate(batches):
                x_batch, y_batch = data.get_batch_data(batch)

                if image_transforms_training:
                    x_batch = self.apply_transforms_to_images(x_batch, image_transforms_training)

                x_batch = torch.stack(x_batch)
                y_batch = torch.Tensor(y_batch).long()

                y_predicted = self.model(x_batch)
                self.loss = self.loss_function(y_predicted, y_batch)
                self.update_weights()
                losses.append(self.loss.item())
                self.log_batch()

            self.loss_valid = self.calculate_validation_loss(x_valid, y_valid)
            validation_losses.append(self.loss_valid.item())
            self.log_epoch()

            if self.epoch > 1 and early_stop_option:
                if self.is_stop_criteria_filled(validation_losses):
                    logger.info('Early stop criterion filled; fitting completed!')
                    return self.model, losses, validation_losses

        return self.model, losses, validation_losses

    def log_epoch(self):
        logger.info(f'''
                        Epoch: {self.epoch+1}/{self.epochs}
                        Validation loss: {self.loss_valid.item()}''')

    def log_batch(self):
        logger.debug(f'''
                Epoch: {self.epoch+1}/{self.epochs}
                Batch: {self.batch_index}
                Training loss: {self.loss.item()}''')

    def calculate_validation_loss(self, x_valid, y_valid):
        with torch.no_grad():
            y_predicted_valid = self.model(x_valid)
            loss_valid = self.loss_function(y_predicted_valid, y_valid)
        return loss_valid

    def update_weights(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    @staticmethod
    def is_stop_criteria_filled(validation_losses):
        return validation_losses[-1] > validation_losses[-2]

    @staticmethod
    def apply_transforms_to_images(images, transforms):
        return [transforms(image) for image in images]
