import logging

import click

from cli_app.classification import classify_images
from cli_app.training import train_model

logging.basicConfig(level=logging.INFO)


@click.group()
def main():
    """
    Simple CLI app for training model and classifying images
    """
    pass


@main.command('classify')
@click.option('--model',
              default='model.p',
              help='Model file path')
@click.option('--source',
              default='data',
              help='Folder path containing images')
@click.option('--target',
              default=None,
              help='Folder path where results will be saved')
@click.option('--image_type',
              default='jpg',
              help='Image extension')
def classify(model, source, target, image_type):
    classify_images(model, source, target, image_type)


@main.command('train')
@click.option('--source',
              default='data',
              help='Folder path containing labeled images, labels as subfolder names')
@click.option('--target',
              default='model.p',
              help='Path where model will be saved')
@click.option('--batch_size',
              default=6,
              help='Batch size for model training')
@click.option('--epochs',
              default=15,
              help='Number of epochs for model training')
@click.option('--dropout',
              default=0.4,
              help='Value of dropout for last classification layer of CNN')
@click.option('--learning_rate',
              default=0.001,
              help='Training learning rate')
@click.option('--weight_decay',
              default=0.01,
              help='Weight decay used for regularization')
@click.option('--early_stop',
              default=True,
              type=bool,
              help='Use early stop in model training')
@click.option('--is_multilabel',
              default=False,
              type=bool,
              help='Is this multi-label classification?')
@click.option('--split_labels_by',
              default=',',
              help='Split folder names to labels using this as separator')
def train(source,
          target,
          batch_size,
          epochs,
          dropout,
          learning_rate,
          weight_decay,
          early_stop,
          is_multilabel,
          split_labels_by):
    train_model(source_data_path=source,
                model_path=target,
                batch_size=batch_size,
                epochs=epochs,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stop_option=early_stop,
                is_multilabel=is_multilabel,
                split_labels_by=split_labels_by)


if __name__ == '__main__':
    main()
