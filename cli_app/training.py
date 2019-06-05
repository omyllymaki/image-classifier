import logging

from data_loaders.image_loader_labels_from_folders import ImageLoaderFromFolders
from file_io import save_pickle_file
from image_data import ImageData
from image_transforms import TransformsTest, TransformsTraining
from interpreters.utils import get_interpreter
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
    data_loader = ImageLoaderFromFolders()
    data = data_loader.load_images_with_labels(source_data_path, split_labels_by=split_labels_by)
    logger.info('Data loading finished')

    image_data = ImageData(data, 0.7, 0.3, 0)
    n_classes = len(image_data.labels)
    model = get_pretrained_model_for_transfer_learning(n_classes, is_multilabel, dropout)
    Learner = get_learner(is_multilabel)
    learner = Learner(model)

    logger.info('Start model training')
    _, _ = learner.fit_model(image_data,
                             image_transforms_training=TransformsTraining,
                             image_transforms_validation=TransformsTest,
                             batch_size=batch_size,
                             epochs=epochs,
                             learning_rate=learning_rate,
                             weight_decay=weight_decay,
                             early_stop_option=early_stop_option)

    save_pickle_file(learner, model_path)
    logger.info('Model training finished')
    logger.info(f'Model saved to path: {model_path}')

    logger.info('Evaluating model performance')
    images = image_data.get_images('validation')
    true_classes = image_data.get_classes('validation')
    predicted_classes, probabilities = learner.predict(images, TransformsTest)

    Interpreter = get_interpreter(is_multilabel)
    interpreter = Interpreter(images, predicted_classes, true_classes, probabilities, learner.class_to_label_mapping)
    accuracy = interpreter.calculate_accuracy()
    accuracy_by_label = interpreter.calculate_accuracy_by_label()
    confusion_matrix = interpreter.calculate_confusion_matrix()

    logger.info(f'Overall accuracy of the model: {accuracy}')
    logger.info(f'Accuracy by label: \n{accuracy_by_label}')
    logger.info(f'Confusion matrix: \n{confusion_matrix}')

