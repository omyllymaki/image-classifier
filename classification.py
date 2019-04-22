import logging
import os
import shutil

from data_loader import DataLoader
from file_io import load_pickle_file
from image_transforms import IMAGE_TRANSFORMS

logger = logging.getLogger(__name__)


def classify_images(model_path: str = 'model.p',
                    source_path: str = 'data',
                    target_path: str = None,
                    file_type: str = 'jpg'
                    ):
    if not target_path:
        target_path = os.path.join(source_path, 'classified images')

    logger.debug(f'Model path: {model_path}')
    logger.debug(f'Source path: {source_path}')
    logger.debug(f'Target path: {target_path}')

    model = load_pickle_file(model_path)

    logger.info('Start image loading')
    loader = DataLoader()
    images = loader.get_image_data(source_path, file_type)

    logger.info('Start image classification')
    predicted_classes, probabilities = model.predict(images, IMAGE_TRANSFORMS['test'])

    logger.info('Copy images to target path')
    file_paths = loader._get_file_paths(source_path, file_type)
    for file_path, predicted_class in zip(file_paths, predicted_classes):
        label = model.class_to_label_mapping[predicted_class]
        path = os.path.join(target_path, label)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(file_path, path)

    logger.info('Image classification finished')
    logger.info(f'Classified images saved to folder: {target_path}')
