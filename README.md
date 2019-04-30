# Photo Classifier

Photo classifier based on pretrained CNN model and transfer learning.

Currently supports only single-label classification.

You can train your own classifier by downloading set of Google images for the target task.

## Downloading Google images

If you want to download more than 100 images per keyword, then 
you will need to download chromedriver. You can get if from here:

https://sites.google.com/a/chromium.org/chromedriver/downloads

Download correct driver and move it to root directory of this repository.

Run download_google_images.py script to get images for classification task. This
will create data folder containing images of specified image classes. By default, the
script will load images of teddybears, grizzly bears and black bears.

## Notebooks

**model_training.ipynb**

Loads image data and trains CNN model. Saves trained model as pickle file.

**model_evaluation.ipynb**

Loads trained model and evaluates results and performance of the model using holdout test set.

## CLI app

Simple CLI app for training model and classifying images.

**Model training**

Usage: cli_app.py train [OPTIONS]

Options:

--source TEXT            Folder path containing labeled images, labels as subfolder names

--target TEXT            Path where model will be saved

--batch_size INTEGER     Batch size for model training

--epochs INTEGER         Number of epochs for model training

--dropout FLOAT          Value of dropout for last classification layer of CNN

--learning_rate FLOAT    Training learning rate

--weight_decay FLOAT     Weight decay used for regularization

--early_stop BOOLEAN     Use early stop in model training

--is_multilabel BOOLEAN  Is this multi-label classification?

--split_labels_by TEXT   Split folder names to labels using this as separator

--help                   Show this message and exit.

**Classification of images**

Usage: cli_app.py classify [OPTIONS]

Options:

--model TEXT            Model file path

--source TEXT           Folder path containing images

--target TEXT           Folder path where results will be saved

--image_type TEXT       Image extension

--help                  Show this message and exit.


