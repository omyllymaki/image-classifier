from torch import nn
from torchvision import models


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def classifier(n_inputs: int, n_outputs: int):
    return nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, n_outputs),
        nn.LogSoftmax(dim=1)
    )


def get_pretrained_vgg16(n_classes: int):
    model = models.vgg16(pretrained=True)
    model = freeze_model_parameters(model)
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = classifier(n_inputs, n_classes)
    return model
