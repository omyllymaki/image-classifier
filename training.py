import logging

from data_loaders.image_loader_labels_from_folders import ImageLoader
from file_io import save_pickle_file
from image_data import ImageData
from image_transforms import IMAGE_TRANSFORMS
from learners.utils import get_learner
from model import get_pretrained_model_for_transfer_learning

logger = logging.getLogger(__name__)


def train_model(source_data_path: str,
                model_path: str,
                batch_size: int,
                epochs: int,
                dropout: float,
                learning_rate: float,
                weight_decay: float,
                early_stop_option: bool,
                is_multilabel: bool,
                split_labels_by: str):
    logger.info('Start data loading')
    data_loader = ImageLoader()
    data = data_loader.load_images_with_labels(source_data_path, split_labels_by=split_labels_by)
    logger.info('Data loading finished')

    image_data = ImageData(data, 0.7, 0.3, 0)
    n_classes = len(image_data.labels)
    model = get_pretrained_model_for_transfer_learning(n_classes, is_multilabel, dropout)
    Learner = get_learner(is_multilabel)
    learner = Learner(model)

    logger.info('Start model training')
    _, _ = learner.fit_model(image_data,
                             image_transforms_training=IMAGE_TRANSFORMS['training'],
                             image_transforms_validation=IMAGE_TRANSFORMS['validation'],
                             batch_size=batch_size,
                             epochs=epochs,
                             learning_rate=learning_rate,
                             weight_decay=weight_decay,
                             early_stop_option=early_stop_option)

    save_pickle_file(learner, model_path)
    logger.info('Model training finished')
    logger.info(f'Model saved to path: {model_path}')
