import os
## Model customization ##

# Information describing your dataframe
TEXT_COL_NAME = "text_clean"
LABEL_LIST = ["skincare"]

# Training variables
LABEL_COL_NAME = "labels"
LABEL_DELIM = ","
OTHER_LABEL_NAME = "other"
PREDICTION_COL_NAME = "predict"

# Training parameters
METRIC_TO_MONITOR = "valid_loss"
RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.25
PREDICTION_THRESHOLD = 0.3


## Filepaths and filenames ##

# Pretrained filenames
LABELLED_DATASET_FILE = "labelled_dataset.csv"
CONFIG_PRETRAINED_FILE = "config.json"
VOCAB_PRETRAINED_FILE = "vocab"
WEIGHTS_PRETRAINED_FILE = "weights"
SP_MODEL_FILE = "spm.model"

# GCS File paths
PRETRAINED_GCS_FOLDER = "pretrained/{}"
LM_FORWARD_FOLDER = "forward"
LM_BACKWARD_FOLDER = "backward"
SPM_FOLDER = "spm"
DATASET_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, LABELLED_DATASET_FILE)
CONFIG_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}", CONFIG_PRETRAINED_FILE)
VOCAB_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}", f"{VOCAB_PRETRAINED_FILE}.pkl")
WEIGHTS_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, "{}", f"{WEIGHTS_PRETRAINED_FILE}.pth")
SP_MODEL_GCS_PATH = os.path.join(PRETRAINED_GCS_FOLDER, SPM_FOLDER, SP_MODEL_FILE)

# Local File paths
MODELS_LOCAL_FOLDER = "models" 
DATASET_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, LABELLED_DATASET_FILE)
CONFIG_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, CONFIG_PRETRAINED_FILE)
VOCAB_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, f"{VOCAB_PRETRAINED_FILE}.pkl")
WEIGHTS_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, f"{WEIGHTS_PRETRAINED_FILE}.pth")
SP_MODEL_LOCAL_PATH = os.path.join(MODELS_LOCAL_FOLDER, SP_MODEL_FILE)

# Trained model filenames
LM_MODEL_PATH = "."
BESTMODEL_NAME = "bestmodel"
BESTMODEL_NAME_UNF = "bestmodel_unf"
ENCODER_FILE_NAME = "encoder"
MODEL_FILE_NAME = "fastai_model.pth"
LABEL_SCORE_FILE_NAME = "label_scores.json"
