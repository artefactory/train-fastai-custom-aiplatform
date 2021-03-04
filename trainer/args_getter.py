import argparse

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments to train the model')
    
    # GCS arguments
    parser.add_argument(
        '--bucket-name',
        default=None,
        help='The name of your bucket in GCS')
    parser.add_argument(
        '--model-dir',
        default=None,
        help='The directory to store the model in GCS')
    parser.add_argument(
        '--model-filename',
        default="fastai_model.pth",
        help='The name given to your model saved in GCS')
    parser.add_argument(
        '--score-filename',
        default="label_scores.json",
        help='The name given to the file containing the score of your model saved in GCS')
    parser.add_argument(
        '--dataset-filename',
        default="labelled_dataset.csv",
        help='The name of the labelled dataset csv file stored in GCS')

    # Training arguments
    parser.add_argument(
        '--lang',
        default='fr',
        help='The language of the corpus')
    parser.add_argument(
        '--drop-mult',
        type=float,
        default=0.3,
        help='The drop multiplier value')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar='N',
        help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=8,
        metavar='N',
        help='number of epochs to train (default: 1)')
    parser.add_argument(
        '--bw',
        default=False,
        action='store_true',
        help='Indicates if the pretrained LM is forward or backward')
    parser.add_argument(
        '--monitored-metric',
        default='valid_loss',
        help='The metric to be monitored to select the best model during the training')
    parser.add_argument(
        '--prediction-threshold',
        type=float,
        default=0.3,
        help='Threshold from which we consider the prediction is positive')
    
    # Labelled dataset information 
    parser.add_argument(
        '--text-col',
        default="text_clean",
        help='Name of the column containing texts in your labelled dataset')
    parser.add_argument(
        '--label-delim',
        default=",",
        help='Character that delimits your various labels')
    parser.add_argument(
        '--label-list',
        default="skincare",
        help='The list of your labels (corresponding to columns) in your dataset, separated by label_delim')
    parser.add_argument(
        '--other-label',
        default="other",
        help='The name to give to the label corresponding to no other labels')
    args = parser.parse_args()
    return args
