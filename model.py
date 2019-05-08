from functools import partial

from torch import nn
from torchvision import models


class UnknownModelArchitectureException(Exception):
    pass


def mlp_classifier(n_inputs: int, n_outputs: int, is_multilabel: bool, dropout: float = 0.4):
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


def prepare_model_for_transfer_learning(model, custom_classifier):
    model = freeze_model_parameters(model)
    model = replace_last_layer_with_custom_classifier(model, custom_classifier)
    return model


def get_pretrained_model_for_transfer_learning(n_classes: int,
                                               is_multilabel: bool,
                                               dropout: float = 0.4,
                                               model_architecture: str = 'vgg16'):
    model = get_pretrained_model(model_architecture)
    custom_classifier = partial(mlp_classifier, n_outputs=n_classes, is_multilabel=is_multilabel, dropout=dropout)
    model = prepare_model_for_transfer_learning(model, custom_classifier)
    return model


def replace_last_layer_with_custom_classifier(model, custom_classifier):
    # TODO: is it possible to handle all model architectures without if-elif-else structure?
    if isinstance(model, (models.VGG, models.AlexNet)):
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = custom_classifier(n_inputs=n_inputs)
    elif isinstance(model, models.ResNet):
        n_inputs = model.fc.in_features
        model.fc = custom_classifier(n_inputs=n_inputs)
    elif isinstance(model, models.DenseNet):
        n_inputs = model.classifier.in_features
        model.classifier = custom_classifier(n_inputs=n_inputs)
    else:
        raise UnknownModelArchitectureException
    return model
