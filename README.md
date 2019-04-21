# Photo Classifier

Photo classifier based on pretrained CNN model and transfer learning.

## CLI app

Simple CLI app for training model and classifying images.

**Model training**

Usage: cli_app.py train [OPTIONS]

Options:
-  -s, --source TEXT         Folder path containing labeled images, labels as subfolder names
-  -t, --target TEXT         Path where model will be saved
-  -b, --batch_size INTEGER  Batch size for model training
-  -e, --epochs INTEGER      Number of epochs for model training
-  --early_stop BOOLEAN      Use early stop in model training
-  --help                    Show this message and exit.

**Classification of images**

Usage: cli_app.py classify [OPTIONS]

Options:
-  -m, --model TEXT          Model file path
-  -s, --source TEXT         Folder path containing images
-  -t, --target TEXT         Folder path where results will be saved
-  -type, --image_type TEXT  Image extension
-  --help                    Show this message and exit.


