import os

from gcs_utils import upload_file_to_gcs

from fastai_train import train_fastai_model
from args_getter import get_args

def train_model():
    # Import args
    args = get_args()

    # Train FastAI model
    model_file_name, label_scores_file_name = train_fastai_model(args)

    # Upload model to GCS
    if args.model_dir:
        upload_file_to_gcs(args.bucket_name, model_file_name, os.path.join(args.model_dir, model_file_name))
        upload_file_to_gcs(args.bucket_name, label_scores_file_name, os.path.join(args.model_dir, label_scores_file_name))


if __name__ == '__main__':
    # Launch training of the model
    train_model()
