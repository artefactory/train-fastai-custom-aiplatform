import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse
import subprocess
import hypertune
from fastai.callback.progress import ShowGraphCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.data.block import MultiCategoryBlock, DataBlock
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.metrics import Perplexity, accuracy, error_rate
from fastai.text.all import (AWD_LSTM, AWD_QRNN, awd_lstm_clas_config, awd_lstm_lm_config, awd_qrnn_clas_config)
from fastai.text.data import TextBlock
from fastai.text.learner import language_model_learner, text_classifier_learner
from fastai.learner import load_learner
import torch
from google.cloud import storage

print(torch.cuda.is_available())

MODEL_FILE_NAME = 'fastai_model.pth'
RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.25

def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=16,
      metavar='N',
      help='input batch size for training (default: 64)')
  parser.add_argument(
      '--epochs',
      type=int,
      default=1,
      metavar='N',
      help='number of epochs to train (default: 10)')
  parser.add_argument(
      '--model-dir',
      default=None,
      help='The directory to store the model')

  args = parser.parse_args()
  return args


def _format_column_multilabels(row, label_list, label_delim, other_label_name="other"):
    multilabel = []
    for label in label_list:
        if row[label]:
            multilabel.append(label)
    if multilabel == []:
        result = other_label_name
    else:
        result = label_delim.join(multilabel)
    return result

def download_file(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def update_classif_config(config):
    config_lm = config.copy()
    clf_config = awd_qrnn_clas_config.copy()
    keys_to_remove = set(config_lm.keys()) - set(clf_config.keys())
    for key in keys_to_remove:
        config_lm.pop(key)
    clf_config.update(config_lm)
    return clf_config


def finetune():
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='valid_loss',
        metric_value="TBM",
        global_step="TBM")


def train_lm(train_df, config, args):
    blocks = TextBlock.from_df("text_clean",
                               is_lm=True)

    data_block = DataBlock(blocks=blocks,
                           get_x=ColReader("text"),
                           splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE))

    lm_dataloaders =(data_block).dataloaders(train_df,
                                         bs=args.batch_size,
                                         backwards = True)

    pretrained_filenames = ["weights", "vocab"]

    learner_lm = language_model_learner(lm_dataloaders,
                                    AWD_QRNN,
                                    config=config,
                                    pretrained=True,
                                    path=".",
                                    pretrained_fnames=pretrained_filenames)

    learner_lm = language_model_learner(lm_dataloaders,
                                    AWD_QRNN,
                                    config=config,
                                    pretrained=True,
                                    path=".",
                                    pretrained_fnames=pretrained_filenames)

    lr, _ = learner_lm.lr_find(suggestions=True)

    learner_lm.fit_one_cycle(args.epochs,
                         lr_max = slice(lr),
                         cbs=[ShowGraphCallback(),
                              SaveModelCallback(monitor="valid_loss", fname="bestmodel")])
    learner_lm.load("bestmodel")
    learner_lm.unfreeze()
    learner_lm.fit_one_cycle(args.epochs,
                         cbs=[ShowGraphCallback(),
                              SaveModelCallback(monitor="valid_loss", fname="bestmodel")])
    learner_lm.load("bestmodel")
    learner_lm.save_encoder("encoder")
    return lm_dataloaders


def train_classifier(train_df, lm_dls, config, args):
    blocks = (TextBlock.from_df("text_clean",
                                seq_len=lm_dls.seq_len,
                                vocab=lm_dls.vocab),
              MultiCategoryBlock())

    clf_datablock = DataBlock(blocks=blocks,
                              get_x=ColReader("text"),
                              get_y=ColReader("labels", label_delim=","),
                              splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE)
                              )

    clf_dataloaders = clf_datablock.dataloaders(train_df,
                                                bs=args.batch_size)

    config_cls = update_classif_config(config)

    learner_clf = text_classifier_learner(clf_dataloaders,
                                      AWD_QRNN,
                                      path=clf_dataloaders.path,
                                      drop_mult=0.3,
                                      config=config_cls)
    learner_clf.load_encoder("encoder")
    lr, _ = learner_clf.lr_find(suggestions=True)

    learner_clf.fit_one_cycle(args.epochs,
                              lr_max = slice(lr),
                              cbs=[ShowGraphCallback(),
                                   SaveModelCallback(monitor="valid_loss", fname="bestmodel")])
    learner_clf.load("bestmodel")
    learner_clf.unfreeze()
    learner_clf.fit_one_cycle(args.epochs,
                              lr_max = slice(lr),
                              cbs=[ShowGraphCallback(),
                                   SaveModelCallback(monitor="valid_loss", fname="bestmodel")])
    learner_clf.load("bestmodel")
    learner_clf.export(MODEL_FILE_NAME)

    if args.model_dir:
        subprocess.check_call([
            'gsutil', 'cp', MODEL_FILE_NAME,
            os.path.join(args.model_dir, MODEL_FILE_NAME)])


def train_model():
    args = get_args()
    download_file("sacha-aiplatform", "pretrained/labelled_dataset_10k.csv", "trainer/labelled_dataset.csv")
    download_file("sacha-aiplatform", "pretrained/vocab.pkl", "models/vocab.pkl")
    download_file("sacha-aiplatform", "pretrained/config.json", "trainer/config.json")
    download_file("sacha-aiplatform", "pretrained/weights.pth", "models/weights.pth")
    df = pd.read_csv("trainer/labelled_dataset.csv")
    df.loc[:, "labels"] = df.apply(_format_column_multilabels, args=(["skincare", "makeup"], ","), axis=1)

    train_df, test_df = train_test_split(df.dropna(), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    with open("trainer/config.json", "r") as config_file:
        config = json.load(config_file)
        config.pop("qrnn")

    lm_dls = train_lm(train_df, config, args)
    train_classifier(train_df, lm_dls, config, args)

if __name__ == '__main__':
  train_model()
