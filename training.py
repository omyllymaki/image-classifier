import logging

from data_loader import DataLoader
from file_io import save_pickle_file
from image_data import ImageData
from image_transforms import IMAGE_TRANSFORMS
from learner import Learner
from model import get_pretrained_vgg16

logger = logging.getLogger(__name__)


def train_model(source_data_path: str = 'data',
                model_path: str = 'model.p',
                batch_size: int = 5,
                epochs: int = 1,
                early_stop_option: bool = True):
    logger.info('Start data loading')
    data_loader = DataLoader()
    data = data_loader.get_labeled_image_data(source_data_path)

    image_data = ImageData(data, 0.7, 0.3, 0)

    n_classes = len(image_data.labels)
    model = get_pretrained_vgg16(n_classes)

    logger.info('Start model training')
    learner = Learner(model)
    losses, losses_valid = learner.fit_model(image_data,
                                             image_transforms_training=IMAGE_TRANSFORMS['training'],
                                             image_transforms_validation=IMAGE_TRANSFORMS['validation'],
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             early_stop_option=early_stop_option)

    save_pickle_file(learner, model_path)

    logger.info('Model training finished')
    logger.info(f'Model saved to path: {model_path}')
