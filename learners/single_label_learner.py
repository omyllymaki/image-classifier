from typing import Callable, List

import numpy as np
import torch

from learners.base_learner import BaseLearner


class SingleLabelLearner(BaseLearner):
    def __init__(self,
                 model,
                 loss_function: Callable = torch.nn.NLLLoss,
                 optimizer_function: Callable = torch.optim.Adam):
        super().__init__(model, loss_function, optimizer_function)

    def classes_to_target_tensor(self, classes_list: List[int]) -> torch.Tensor:
        classes_array_flattened = np.array(classes_list).flatten()
        return torch.Tensor(classes_array_flattened).long()

    def get_predicted_classes(self, probabilities) -> List[int]:
        predicted_class = np.argmax(probabilities, axis=0)
        return [int(predicted_class)]
