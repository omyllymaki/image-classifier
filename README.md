# Image Classifier

Image classifier based on pretrained CNN model and transfer learning.

Supports both single-label and multi-label classification.


## Installation

- Create virtual environment

```
virtualenv venv
```

- Activate virtual environment

```
.\venv\scripts\activate     (Windows)
source venv/bin/activate    (Linux)
```

- Install requirements

```
pip install -r requirements.txt
```

Note: to install specific version of PyTorch, use installation instructions found from 

https://pytorch.org/

- Install new kernel

```
ipython kernel install --user --name=image-classifier
```

- Change kernel of notebooks to image-classifier

## Basic usage

**First try**

Repo contains some test data (mugs / broken mugs / filled mugs). Train classifier to recognize 
mugs by running model_training.ipynb notebook. After this, you can evaluate results by 
running model_evaluation.ipynb. You should get about about 90 % accuracy with default parameters.

**Train your own classifier for the target problem**

- Get you own data set by running download_google_images.py script (more detailed instructions below)
- Arrange data so that images are grouped to folders with label name as folder name
- Train and evaluate classifier by running model_training and model_evaluation notebooks
- Improve classifier accuracy by changing constant values in constants.py file. You can also try 
hyperparameter tuning with hyperparameter_tuning notebook.


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

**hyperparameter_tuning.ipynb**

Testing and optimization of model hyperparameters.

**image_labeling.ipynb**

GUI tool for image labeling. Supports guided labeling (active learning).


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


