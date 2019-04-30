import logging

from data_loader import DataLoader
from file_io import save_pickle_file
from image_data import ImageData
from image_transforms import IMAGE_TRANSFORMS
from learner import MultiLabelLearner, SingleLabelLearner
from model import get_pretrained_vgg16

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
    if is_multilabel:
        Learner = MultiLabelLearner
    else:
        Learner = SingleLabelLearner

    logger.info('Start data loading')
    data_loader = DataLoader()
    data = data_loader.get_labeled_image_data(source_data_path, split_labels_by=split_labels_by)
    logger.info('Data loading finished')

    image_data = ImageData(data, 0.7, 0.3, 0)
    n_classes = len(image_data.labels)
    model = get_pretrained_vgg16(n_classes, is_multilabel, dropout)
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
