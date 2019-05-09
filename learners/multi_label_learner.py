from typing import List

import numpy as np
import torch

from learners.base_learner import BaseLearner


class MultiLabelLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, loss_function=torch.nn.MultiLabelSoftMarginLoss, **kwargs)

    def classes_to_target_tensor(self, classes_list: List[int]) -> torch.Tensor:
        classes_one_hot_encoded = self.one_hot_encoder.fit_transform(classes_list)
        return torch.Tensor(classes_one_hot_encoded)

    def get_predicted_classes(self, probabilities, threshold: float = 0.5) -> List[int]:
        predicted_classes = np.where(probabilities > threshold)
        return predicted_classes[0].tolist()
