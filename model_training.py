import logging
from random import shuffle

import torch

logger = logging.getLogger(__name__)


def make_batches(items, batch_size=1, shuffle_option=True):
    if shuffle_option:
        shuffle(items)
    length = len(items)
    for index in range(0, length, batch_size):
        yield items[index:min(index + batch_size, length)]


def apply_transforms_to_images(images, transforms):
    return [transforms(image) for image in images]


def update_weights(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def fit_model(data,
              model,
              batch_size=1,
              epochs=1,
              image_transforms=None,
              loss_function=None,
              optimizer=None,
              early_stop_option=True):
    if not loss_function:
        loss_function = torch.nn.NLLLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    training_data = data.training_data
    validation_data = data.validation_data
    x_valid = [d['x'] for d in validation_data]
    y_valid = [d['y'] for d in validation_data]
    indices = list(range(len(training_data)))

    if image_transforms:
        x_valid = apply_transforms_to_images(x_valid, image_transforms['validation'])
    x_valid = torch.stack(x_valid)
    y_valid = torch.Tensor(y_valid).long()

    losses, validation_losses = [], []
    for epoch in range(epochs):
        batches = make_batches(indices, batch_size=batch_size)
        number_of_batches = int(len(indices) / batch_size)

        for batch_index, batch in enumerate(batches):
            batch_data = [training_data[i] for i in batch]
            x_batch = [d['x'] for d in batch_data]
            y_batch = [d['y'] for d in batch_data]

            if image_transforms:
                x_batch = apply_transforms_to_images(x_batch, image_transforms['training'])

            x_batch = torch.stack(x_batch)
            y_batch = torch.Tensor(y_batch).long()

            y_predicted = model(x_batch)
            loss = loss_function(y_predicted, y_batch)
            update_weights(optimizer, loss)

            losses.append(loss.item())

            logger.debug(f'''
            Epoch: {epoch+1}/{epochs}
            Batch: {batch_index}/{number_of_batches}
            Training loss: {loss.item()}''')

        with torch.no_grad():
            y_predicted_valid = model(x_valid)
            loss_valid = loss_function(y_predicted_valid, y_valid)
        validation_losses.append(loss_valid)
        logger.info(f'''
                    Epoch: {epoch+1}/{epochs}
                    Validation loss: {loss_valid.item()}''')

        if epoch > 1 and early_stop_option:
            if validation_losses[-1] > validation_losses[-2]:
                logger.info('Early stop criterion filled; fitting completed!')
                return model, losses, validation_losses

    return model, losses, validation_losses
