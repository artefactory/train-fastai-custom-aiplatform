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

MODEL_FILE_NAME = 'model.pth'

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


def download_labelled_dataset(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def upload_trained_model(bucket_name, model_filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_filename)
    print(
        "Blob {} uploaded to {}.".format(
            model_filename, bucket_name
        )
    )


def train_model():
    args = get_args()
    download_labelled_dataset("sacha-aiplatform", "datasets/en_labelled_dataset.csv", "trainer/labelled_dataset.csv")

    df = pd.read_csv("trainer/labelled_dataset.csv")
    df.loc[:, "labels"] = df.apply(_format_column_multilabels, args=(["skincare", "makeup", "hairmanagement"], ","), axis=1)
    RANDOM_STATE = 42
    VAL_SIZE = 0.2
    TEST_SIZE = 0.25

    train_df, test_df = train_test_split(df.dropna(), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    blocks = (TextBlock.from_df("text_clean"), MultiCategoryBlock())

    clf_datablock = DataBlock(blocks=blocks,
                              get_x=ColReader("text"),
                              get_y=ColReader("labels", label_delim=","),
                              splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE)
                              )

    clf_dataloaders = clf_datablock.dataloaders(train_df,
                                                bs=args.batch_size)

    learner_clf = text_classifier_learner(clf_dataloaders,
                                      AWD_LSTM)

    learner_clf.fit_one_cycle(args.epochs)
    learner_clf.export(MODEL_FILE_NAME)

    #upload_trained_model("sacha-aiplatform", MODEL_FILE_NAME)
    if args.model_dir:
        subprocess.check_call([
            'gsutil', 'cp', MODEL_FILE_NAME,
            os.path.join(args.model_dir, MODEL_FILE_NAME)])



if __name__ == '__main__':
  train_model()
