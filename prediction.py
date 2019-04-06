from typing import Tuple

import numpy as np
import torch


def predict(images: list, model, transforms=None) -> Tuple[np.ndarray, np.ndarray]:
    predicted_classes, probabilities = [], []
    for image in images:
        if transforms:
            image = transforms(image)
        image = torch.Tensor(image).unsqueeze(0)
        prediction = model(image)
        prob = torch.exp(prediction).detach().numpy()[0]
        predicted_class = np.argmax(prob, axis=0)
        predicted_classes.append(predicted_class)
        probabilities.append(prob)
    return np.array(predicted_classes), np.array(probabilities)


def calculate_accuracy(values_true: np.ndarray, values_predicted: np.ndarray) -> float:
    return sum(values_predicted == values_true) / len(values_true)
