import logging

from constants import SOURCE_DATA_PATH, P_TRAINING, P_VALID, P_TEST, BATCH_SIZE, EPOCHS, MODEL_FILE_PATH
from data_loader import DataLoader
from file_io import save_pickle_file
from image_data import ImageData
from image_transforms import IMAGE_TRANSFORMS
from learner import Learner
from model import get_pretrained_vgg16

logging.basicConfig(level=logging.INFO)
data_loader = DataLoader()
data = data_loader.get_labeled_data(SOURCE_DATA_PATH)

image_data = ImageData(data, P_TRAINING, P_VALID, P_TEST)

n_classes = len(image_data.labels)
model = get_pretrained_vgg16(n_classes)

learner = Learner(model)
model, losses, losses_valid = learner.fit_model(image_data,
                                                batch_size=BATCH_SIZE,
                                                epochs=EPOCHS,
                                                image_transforms=IMAGE_TRANSFORMS,
                                                early_stop_option=True)

data = {
    'model': model,
    'data': image_data,
}
save_pickle_file(data, MODEL_FILE_PATH)
