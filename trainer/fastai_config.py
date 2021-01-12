import os
## Model customization ##

# Information on the DataFrame
TEXT_COL_NAME = "text_clean"
LABEL_COL_NAME = "labels"
LABEL_DELIM = ","
LABEL_LIST = ["skincare"]
OTHER_LABEL_NAME = "other"

# Model parameters
METRIC_TO_MONITOR = "valid_loss"

# Variables of train/test split
RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.25


## Filepaths and filenames ##

# GCS File paths
PRETRAINED_GCS_FOLDER = "pretrained/{}"
# TBM -> Need to be reorganized
DATASET_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "labelled_dataset.csv")
VOCAB_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}", "vocab.pkl")
CONFIG_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}","config.json")
WEIGHTS_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}","weights.pth")

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
LM_MODEL_PATH = "."
BESTMODEL_NAME = "bestmodel"
ENCODER_FILE_NAME = "encoder"
MODEL_FILE_NAME = "fastai_model.pth"
