import os
import argparse

from gcs_utils import upload_file_to_gcs

from fastai_train import train_fastai_model


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments to train the model')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar='N',
        help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        metavar='N',
        help='number of epochs to train (default: 1)')
    parser.add_argument(
        '--bucket-name',
        default=None,
        help='The name of your bucket  in your GCP project')
    parser.add_argument(
        '--model-dir',
        default=None,
        help='The directory to store the model')
    args = parser.parse_args()
    return args

def train_model():
    # Import args
    args = get_args()

    # Train FastAI model
    model_file_name = train_fastai_model(args)

    # Upload model to GCS
    if args.model_dir:
        gcs_file_path = os.path.join(f'gs://{args.bucket_name}', args.model_dir, model_file_name)
        upload_file_to_gcs(model_file_name,
                           gcs_file_path)

if __name__ == '__main__':
    # Launch training of the model
    train_model()
