import pandas as pd

from labeler.recommender import Recommender


class RecommenderSystem:

    def __init__(self, labels, learner):
        self.labels = labels
        self.idx_to_label = {idx: label for idx, label in enumerate(self.labels)}
        self.label_to_idx = {label: idx for idx, label in self.idx_to_label.items()}
        self.recommender = Recommender(learner)

    def train(self):
        images_with_labels = self.get_images_with_labels()
        self.recommender.train_model(images_with_labels)

    def query(self, df: pd.DataFrame):
        self.df = df
        self.train()
        unlabeled_images = self.get_unlabeled_images()
        indices = self.recommender.query(unlabeled_images)
        return indices

    def get_images_with_labels(self):
        labeled = self._get_labeled()
        labeled['y'] = labeled.labels.apply(lambda x: [self.label_to_idx[i] for i in x])
        data = labeled.apply(lambda x: {'x': x.image, 'y': x.y}, axis=1).tolist()
        return data

    def get_unlabeled_images(self):
        unlabeled = self._get_unlabeled()
        return unlabeled.image.tolist()

    def _get_labeled(self):
        return self.df[~self.df.labels.isnull()]

    def _get_unlabeled(self):
        return self.df[self.df.labels.isnull()]
