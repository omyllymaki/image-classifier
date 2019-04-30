from typing import Callable, List

import numpy as np
import torch

from learners.base_learner import BaseLearner


class MultiLabelLearner(BaseLearner):
    def __init__(self,
                 model,
                 loss_function: Callable = torch.nn.MultiLabelSoftMarginLoss,
                 optimizer_function: Callable = torch.optim.Adam):
        super().__init__(model, loss_function, optimizer_function)

    def classes_to_target_tensor(self, classes_list: List[int]) -> torch.Tensor:
        classes_one_hot_encoded = self.one_hot_encoder.fit_transform(classes_list)
        return torch.Tensor(classes_one_hot_encoded)

    def get_predicted_classes(self, probabilities, threshold: float = 0.5) -> List[int]:
        predicted_classes = np.where(probabilities > threshold)
        return predicted_classes[0].tolist()
