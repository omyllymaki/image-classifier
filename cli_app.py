import logging

import click

from classification import classify_images
from training import train_model

logging.basicConfig(level=logging.INFO)


@click.group()
def main():
    """
    Simple CLI app for training model and classifying images
    """
    pass


@main.command('classify')
@click.option('--model', '-m',
              default='model.p',
              help='Model file path')
@click.option('--source', '-s',
              default='data',
              help='Folder path containing images')
@click.option('--target', '-t',
              default=None,
              help='Folder path where results will be saved')
@click.option('--image_type', '-type',
              default='jpg',
              help='Image extension')
def classify(model, source, target, image_type):
    classify_images(model, source, target, image_type)


@main.command('train')
@click.option('--source', '-s',
              default='data',
              help='Folder path containing labeled images, labels as subfolder names')
@click.option('--target', '-t',
              default='model.p',
              help='Path where model will be saved')
@click.option('--batch_size', '-b',
              default=6,
              help='Batch size for model training')
@click.option('--epochs', '-e',
              default=15,
              help='Number of epochs for model training')
@click.option('--early_stop',
              default=True,
              type=bool,
              help='Use early stop in model training')
def train(source, target, batch_size, epochs, early_stop):
    train_model(source, target, batch_size, epochs, early_stop)


if __name__ == '__main__':
    main()
