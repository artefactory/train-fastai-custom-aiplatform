import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

from fastai.callback.progress import ShowGraphCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.data.block import MultiCategoryBlock, CategoryBlock, DataBlock
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.metrics import Perplexity, accuracy, error_rate
from fastai.text.all import (AWD_LSTM, AWD_QRNN, awd_lstm_clas_config, awd_qrnn_lm_config, awd_qrnn_clas_config)
from fastai.text.data import TextBlock
from fastai.text.learner import language_model_learner, text_classifier_learner
from fastai.learner import load_learner
import torch

from fastai_config import (DATASET_GCS_PATH, VOCAB_GCS_PATH, CONFIG_GCS_PATH, WEIGHTS_GCS_PATH, MODELS_LOCAL_FOLDER, TRAINER_LOCAL_FOLDER,
                    DATASET_LOCAL_PATH, VOCAB_LOCAL_PATH, CONFIG_LOCAL_PATH, WEIGHTS_LOCAL_PATH, RANDOM_STATE, VAL_SIZE, TEST_SIZE,
                    TEXT_COL_NAME, LABEL_COL_NAME, LABEL_LIST, OTHER_LABEL_NAME, LANGUAGE, BESTMODEL_NAME, MODEL_FILE_NAME, LABEL_DELIM,
                    WEIGHTS_PRETRAINED_FILE, VOCAB_PRETRAINED_FILE, DROP_MULT, LM_MODEL_PATH, METRIC_TO_MONITOR, BESTMODEL_NAME, LANGUAGE, MULTICATEGORY)

from gcs_utils import download_file_from_gcs

def _format_column_multilabels(row, label_list, label_delim, other_label_name=OTHER_LABEL_NAME):
    # Format dataframe label column
    multilabel = []
    for label in label_list:
        if row[label]:
            multilabel.append(label)
    if multilabel == []:
        result = other_label_name
    else:
        result = label_delim.join(multilabel)
    return result


def update_classif_config(config):
    # Update LM config file to be used to create a classifier
    config_lm = config.copy()
    clf_config = awd_qrnn_clas_config.copy()
    keys_to_remove = set(config_lm.keys()) - set(clf_config.keys())
    for key in keys_to_remove:
        config_lm.pop(key)
    clf_config.update(config_lm)
    return clf_config


def fit_with_gradual_unfreezing(learner, epochs, lr, cycles):
    learner.fit_one_cycle(epochs,
                          lr_max = slice(lr),
                          cbs=[ShowGraphCallback(),
                               SaveModelCallback(monitor=METRIC_TO_MONITOR, fname=BESTMODEL_NAME)])
    learner.load(BESTMODEL_NAME)
    learner.unfreeze()
    learner.fit_one_cycle(epochs,
                          cbs=[ShowGraphCallback(),
                               SaveModelCallback(monitor=METRIC_TO_MONITOR, fname=BESTMODEL_NAME)])
    learner.load(BESTMODEL_NAME)
    return learner


def find_best_lr(learner):
    lr, _ = learner.lr_find(suggestions=True)
    return lr


def train_lm(train_df, config, args):
    # Function to fine-tune the pre-trained language model
    blocks = TextBlock.from_df(TEXT_COL_NAME,
                               is_lm=True)

    data_block = DataBlock(blocks=blocks,
                           get_x=ColReader("text"),
                           splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE))

    lm_dataloaders =(data_block).dataloaders(train_df,
                                         bs=args.batch_size,
                                         backwards = True)

    pretrained_filenames = [WEIGHTS_PRETRAINED_FILE, VOCAB_PRETRAINED_FILE]

    learner_lm = language_model_learner(lm_dataloaders,
                                        AWD_QRNN,
                                        config=config,
                                        pretrained=True,
                                        path=LM_MODEL_PATH,
                                        pretrained_fnames=pretrained_filenames)

    lr = find_best_lr(learner_lm)

    learner_lm = fit_with_gradual_unfreezing(learner_lm, args.epochs, lr)
    learner_lm.save_encoder("encoder")
    return lm_dataloaders


def train_classifier(train_df, lm_dls, config, args):
    if MULTICATEGORY:
        block_category = MultiCategoryBlock()
    else:
        block_category = CategoryBlock()

    blocks = (TextBlock.from_df(TEXT_COL_NAME,
                                seq_len=lm_dls.seq_len,
                                vocab=lm_dls.vocab),
              block_category)

    clf_datablock = DataBlock(blocks=blocks,
                              get_x=ColReader("text"),
                              get_y=ColReader(LABEL_COL_NAME, label_delim=LABEL_DELIM),
                              splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE)
                              )

    clf_dataloaders = clf_datablock.dataloaders(train_df,
                                                bs=args.batch_size)

    config_cls = update_classif_config(config)

    learner_clf = text_classifier_learner(clf_dataloaders,
                                          AWD_QRNN,
                                          path=clf_dataloaders.path,
                                          drop_mult=DROP_MULT,
                                          config=config_cls)
    learner_clf.load_encoder("encoder")

    lr = find_best_lr(learner_clf)
    learner_clf = fit_with_gradual_unfreezing(learner_clf, args.epochs, lr)
    learner_clf.export(MODEL_FILE_NAME)
    return MODEL_FILE_NAME


def open_config_file(language, config_local_path=None):
    if language == "en" or config_local_path is None:
        config = awd_qrnn_lm_config.copy()
    else:
        with open(config_local_path, "r") as config_file:
            config = json.load(config_file)
            config.pop("qrnn")
    return config


def train_fastai_model(args):
    # Ensure CUDA is enabled
    print(torch.cuda.is_available())

    # Download necessary files (for english language, only the labelled dataset is necessary)
    download_file_from_gcs(args.bucket_name, DATASET_GCS_PATH, DATASET_LOCAL_PATH)
    if LANGUAGE != 'en':
        download_file_from_gcs(args.bucket_name, VOCAB_GCS_PATH, VOCAB_LOCAL_PATH)
        download_file_from_gcs(args.bucket_name, CONFIG_GCS_PATH, CONFIG_LOCAL_PATH)
        download_file_from_gcs(args.bucket_name, WEIGHTS_GCS_PATH, WEIGHTS_LOCAL_PATH)

    # Format dataframe for training
    df = pd.read_csv(DATASET_LOCAL_PATH)
    df.loc[:, LABEL_COL_NAME] = df.apply(_format_column_multilabels, args=(LABEL_LIST, LABEL_DELIM), axis=1)
    train_df, test_df = train_test_split(df.dropna(), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Open json config file
    config = open_config_file(LANGUAGE, CONFIG_LOCAL_PATH)

    # Fine-tune language model
    lm_dls = train_lm(train_df, config, args)

    # Create and train classifier
    model_file_name = train_classifier(train_df, lm_dls, config, args)

    return model_file_name