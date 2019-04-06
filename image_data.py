from random import shuffle
from typing import List


class ImageData:

    def __init__(self,
                 image_data: List[dict],
                 p_training: float = 0.5,
                 p_valid: float = 0.3
                 ):
        self.image_data = image_data
        self.get_label_mappings()
        self.convert_labels_to_integers()
        self.divide_data_to_sets(p_training, p_valid)

    def convert_labels_to_integers(self):
        for item in self.image_data:
            item['y'] = self.label_to_class_mapping[item['y']]

    def get_label_mappings(self):
        self.labels = list(set([sample['y'] for sample in self.image_data]))
        self.label_to_class_mapping = {label: idx for idx, label in enumerate(self.labels)}
        self.class_to_label_mapping = {v: k for k, v in self.label_to_class_mapping.items()}

    def divide_data_to_sets(self, p_training: float, p_validation: float):
        n_images = len(self.image_data)
        indices = list(range(n_images))
        shuffle(indices)

        n_train = int(p_training * n_images)
        n_validation = int(p_validation * n_images)

        indices_train = indices[:n_train]
        indices_validation = indices[n_train:n_train + n_validation]
        indices_test = indices[n_train + n_validation:]

        self.training_data = [self.image_data[i] for i in indices_train]
        self.validation_data = [self.image_data[i] for i in indices_validation]
        self.test_data = [self.image_data[i] for i in indices_test]

    def get_image(self, index, data_set_name):
        data = self.get_data_set(data_set_name)
        return self._get_image(data, index)

    def get_label(self, index, data_set_name):
        data = self.get_data_set(data_set_name)
        return self._get_label(data, index)

    def get_images(self, data_set_name):
        data = self.get_data_set(data_set_name)
        return self._get_images(data)

    def get_labels(self, data_set_name):
        data = self.get_data_set(data_set_name)
        return self._get_labels(data)

    def get_classes(self, data_set_name):
        data = self.get_data_set(data_set_name)
        return self._get_classes(data)

    def get_data_set(self, data_set_name):
        options = {
            'training': self.training_data,
            'validation': self.validation_data,
            'test': self.test_data
        }
        return options.get(data_set_name)

    def convert_classes_to_labels(self, classes):
        return [self.class_to_label_mapping[c] for c in classes]

    def _get_labels(self, data):
        classes = self._get_classes(data)
        return [self.class_to_label_mapping[c] for c in classes]

    def _get_label(self, data, index):
        return self.class_to_label_mapping[data[index]['y']]

    @staticmethod
    def _get_images(data):
        return [sample['x'] for sample in data]

    @staticmethod
    def _get_classes(data):
        return [sample['y'] for sample in data]

    @staticmethod
    def _get_image(data, index):
        return data[index]['x']
