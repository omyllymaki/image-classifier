from torch import nn


def classifier(n_inputs, n_outputs):
    return nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, n_outputs),
        nn.LogSoftmax(dim=1)
    )
