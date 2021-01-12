import os
## Model customization ##

# Information on the DataFrame
TEXT_COL_NAME = "text_clean"
LABEL_COL_NAME = "labels"
LABEL_DELIM = ","
LABEL_LIST = ["skincare", "makeup"]
OTHER_LABEL_NAME = "other"

# Model parameters
METRIC_TO_MONITOR = "valid_loss"
#DROP_MULT = 0.3 # Between 0 and 1
#LANGUAGE = 'fr' # Choose among ['en', 'fr', 'ko', 'ja', 'zh']
BACKWARD = True 

# Variables of train/test split
RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.25


## Filepaths and filenames ##

# GCS File paths
PRETRAINED_GCS_FOLDER = "pretrained/{}"
# TBM -> Need to be reorganized
LM_TYPE_GCS_FOLDER = {False: 'pretrained_language_models/forward',
                      True: 'pretrained_language_models/backward'}
DATASET_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "labelled_dataset.csv")
VOCAB_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, LM_TYPE_GCS_FOLDER.get(BACKWARD), "vocab.pkl")
CONFIG_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, LM_TYPE_GCS_FOLDER.get(BACKWARD), "config.json")
WEIGHTS_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, LM_TYPE_GCS_FOLDER.get(BACKWARD), "weights.pth")

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
