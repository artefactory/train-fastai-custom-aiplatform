import argparse

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments to train the model')
    
    # GCS location arguments
    parser.add_argument(
        '--bucket-name',
        default=None,
        help='The name of your bucket  in your GCP project')
    parser.add_argument(
        '--model-dir',
        default=None,
        help='The directory to store the model')

    # Model arguments
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
        default=1,
        metavar='N',
        help='number of epochs to train (default: 1)')
    parser.add_argument(
        '--bw',
        default=True,
        action='store_true',
        help='Indicates if the pretrained LM is forward or backward')
    args = parser.parse_args()
    return args