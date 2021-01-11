import os

## Filepaths and filenames ##

# GCS File paths
DATASET_GCS_PATH = "pretrained/labelled_dataset_10k.csv"
VOCAB_GCS_PATH = "pretrained/vocab.pkl"
CONFIG_GCS_PATH = "pretrained/config.json"
WEIGHTS_GCS_PATH = "pretrained/weights.pth"

# Local File paths
MODELS_LOCAL_FOLDER = "models"
TRAINER_LOCAL_FOLDER = "trainer"
DATASET_LOCAL_PATH = os.path.join(TRAINER_LOCAL_FOLDER, "labelled_dataset.csv")
VOCAB_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, "vocab.pkl")
CONFIG_LOCAL_PATH = os.path.join(TRAINER_LOCAL_FOLDER, "config.json")
WEIGHTS_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, "weights.pth")

# Pretrained filenames
WEIGHTS_PRETRAINED_FILE = "weights"
VOCAB_PRETRAINED_FILE = "vocab"

# Trained model filename
MODEL_FILE_NAME = "fastai_model.pth"


## Model customization ##

# Information on the DataFrame
TEXT_COL_NAME = "text_clean"
LABEL_COL_NAME = "labels"
LABEL_DELIM = ","
LABEL_LIST = ["skincare", "makeup"]
OTHER_LABEL_NAME = "other"

# Model parameters
LANGUAGE = "fr"
BESTMODEL_NAME = "bestmodel"
METRIC_TO_MONITOR = "valid_loss"
LM_MODEL_PATH = "."
DROP_MULT = 0.3
MULTICATEGORY = True
LANGUAGE = 'fr'

# Variables of train/test split
RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.25





