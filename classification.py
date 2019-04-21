import logging
import os
import shutil

from PIL import Image

from file_io import load_pickle_file
from image_transforms import IMAGE_TRANSFORMS
from utils import get_file_paths

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
    file_paths = get_file_paths(source_path, file_type)

    logger.info('Start image loading')
    images = load_images(file_paths)

    logger.info('Start image classification')
    predicted_classes, probabilities = model.predict(images, IMAGE_TRANSFORMS['test'])

    logger.info('Copy images to target path')
    for file_path, predicted_class in zip(file_paths, predicted_classes):
        label = model.class_to_label_mapping[predicted_class]
        path = os.path.join(target_path, label)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(file_path, path)

    logger.info('Image classification finished')
    logger.info(f'Classified images saved to folder: {target_path}')


def load_images(file_paths):
    n_files = len(file_paths)
    images = []
    for index, file_path in enumerate(file_paths, 1):
        logger.info(f'{index}/{n_files} images loaded')
        image = Image.open(file_path)
        images.append(image.copy())
        image.close()
    return images
