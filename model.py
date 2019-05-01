from torch import nn
from torchvision import models


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


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_pretrained_model(model_architecture: str):
    model = getattr(models, model_architecture)
    return model(pretrained=True)


def prepare_model_for_transfer_learning(model, n_classes: int, is_multilabel: bool, dropout: float = 0.4):
    model = freeze_model_parameters(model)
    n_inputs = model.classifier[-1].in_features
    model.classifier[-1] = classifier(n_inputs, n_classes, is_multilabel, dropout)
    return model


def get_pretrained_model_for_transfer_learning(n_classes: int,
                                               is_multilabel: bool,
                                               dropout: float = 0.4,
                                               model_architecture: str = 'vgg16'):
    model = get_pretrained_model(model_architecture)
    model = prepare_model_for_transfer_learning(model, n_classes, is_multilabel, dropout)
    return model
