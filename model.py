from torch import nn
from torchvision import models


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def classifier(n_inputs: int, n_outputs: int, is_multilabel: bool, dropout: float = 0.4):
    if is_multilabel:
        last_activation_layer = nn.LogSigmoid()
    else:
        last_activation_layer = nn.LogSoftmax(dim=1)
    net = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, n_outputs),
        last_activation_layer
    )
    return net


def get_pretrained_vgg16(n_classes: int, is_multilabel: bool, dropout: float = 0.4):
    model = models.vgg16(pretrained=True)
    model = freeze_model_parameters(model)
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = classifier(n_inputs, n_classes, is_multilabel, dropout)
    return model
