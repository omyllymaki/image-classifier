import matplotlib.pyplot as plt
import numpy as np


def imshow_tensor(image, ax=None, mean=0, std=1):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


def visualize_prediction(image, true_label, labels_predicted, probabilities_predicted):
    plt.subplot(1,2,1)
    plt.barh(labels_predicted, probabilities_predicted)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(image))
    plt.title(f' True label: {true_label}')