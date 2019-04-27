from math import floor
from random import shuffle
from typing import List


class ImageData:

    def __init__(self,
                 image_data: List[dict],
                 p_training: float = 0.5,
                 p_valid: float = 0.25,
                 p_test: float = 0.25,
                 ):
        self.image_data = image_data
        self.get_label_mappings()
        self.convert_labels_to_integers()
        self.divide_data_to_sets(p_training, p_valid, p_test)

    def make_batches(self, data_set_name, batch_size):
        self.data = self.get_data_set(data_set_name)
        indices = list(range(len(self.data)))
        batches = self._make_batches(indices, batch_size=batch_size)
        return batches

    def get_batch_data(self, batch):
        batch_data = [self.data[i] for i in batch]
        batch_images = [d['x'] for d in batch_data]
        batch_classes = [d['y'] for d in batch_data]
        return batch_images, batch_classes

    def convert_labels_to_integers(self):
        image_data = []
        for item in self.image_data:
            x = item['x']
            y = [self.label_to_class_mapping[label] for label in item['y']]
            image_data.append(dict(x=x, y=y))
        self.image_data = image_data

    def get_label_mappings(self):
        all_labels = [sample['y'] for sample in self.image_data]
        self.labels = list(set(x for l in all_labels for x in l))
        self.label_to_class_mapping = {label: idx for idx, label in enumerate(self.labels)}
        self.class_to_label_mapping = {v: k for k, v in self.label_to_class_mapping.items()}

    def divide_data_to_sets(self, p_training: float, p_validation: float, p_test: float):
        n_images = len(self.image_data)
        indices = list(range(n_images))
        shuffle(indices)

        n_train = floor(p_training * n_images)
        n_validation = floor(p_validation * n_images)
        n_test = floor(p_test * n_images)

        indices_train = indices[:n_train]
        indices_validation = indices[n_train:n_train + n_validation]
        indices_test = indices[n_train + n_validation:n_train + n_validation + n_test]

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
        return [self.class_to_label_mapping[item] for item in data[index]['y']]

    @staticmethod
    def _get_images(data):
        return [sample['x'] for sample in data]

    @staticmethod
    def _get_classes(data):
        return [sample['y'] for sample in data]

    @staticmethod
    def _get_image(data, index):
        return data[index]['x']

    @staticmethod
    def _make_batches(items, batch_size=1, shuffle_option=True):
        if shuffle_option:
            shuffle(items)
        length = len(items)
        for index in range(0, length, batch_size):
            yield items[index:min(index + batch_size, length)]
