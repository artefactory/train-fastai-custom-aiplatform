import json

import pandas as pd
import torch
from fastai.callback.progress import ShowGraphCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.data.block import CategoryBlock, DataBlock, MultiCategoryBlock
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.text.all import (AWD_LSTM, AWD_QRNN, awd_lstm_clas_config,
                             awd_lstm_lm_config)
from fastai.text.core import SentencePieceTokenizer
from fastai.text.data import TextBlock
from fastai.text.learner import language_model_learner, text_classifier_learner
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from fastai_config import (BESTMODEL_NAME, BESTMODEL_NAME_UNF, CONFIG_GCS_PATH,
                           CONFIG_LOCAL_PATH, DATASET_GCS_PATH,
                           DATASET_LOCAL_PATH, ENCODER_FILE_NAME,
                           LABEL_COL_NAME,
                           LM_BACKWARD_FOLDER, LM_FORWARD_FOLDER,
                           LM_MODEL_PATH, PREDICTION_COL_NAME, RANDOM_STATE,
                           SP_MODEL_GCS_PATH, SP_MODEL_LOCAL_PATH, TEST_SIZE,
                           VAL_SIZE, VOCAB_GCS_PATH, VOCAB_LOCAL_PATH,
                           VOCAB_PRETRAINED_FILE, WEIGHTS_GCS_PATH,
                           WEIGHTS_LOCAL_PATH, WEIGHTS_PRETRAINED_FILE)
from gcs_utils import download_file_from_gcs


def _format_column_multilabels(row, label_list, label_delim, other_label_name):
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


def open_config_file(language, config_local_path=None):
    # Open config.json file containing model pretrained LM configuration
    if language == "en" or config_local_path is None:
        config = awd_lstm_lm_config.copy()
        return config, AWD_LSTM
    else:
        with open(config_local_path, "r") as config_file:
            config = json.load(config_file)
        if config.get("qrnn") is not None:
            if config.pop("qrnn"):
                return config, AWD_QRNN
            else:
                return config, AWD_LSTM
        return config, AWD_QRNN


def update_classif_config(config):
    # Update LM config file to be used by the classifier dataloaders
    config_lm = config.copy()
    clf_config = awd_lstm_clas_config.copy()
    keys_to_remove = set(config_lm.keys()) - set(clf_config.keys())
    for key in keys_to_remove:
        config_lm.pop(key)
    return config_lm


def find_best_lr(learner):
    # FastAI native method to find the best LR for the training
    lr, _ = learner.lr_find(suggestions=True)
    return lr


def fit_with_gradual_unfreezing(learner, epochs, lr, args):
    # Fit the frozen model on one cycle, unfreeze it, and fit it again on another cycle
    learner.fit_one_cycle(epochs,
                          lr_max = slice(lr),
                          cbs=[ShowGraphCallback(),
                               SaveModelCallback(monitor=args.monitored_metric, fname=BESTMODEL_NAME)])
    learner.load(BESTMODEL_NAME)
    learner.unfreeze()
    learner.fit_one_cycle(epochs,
                          cbs=[ShowGraphCallback(),
                               SaveModelCallback(monitor=args.monitored_metric, fname=BESTMODEL_NAME_UNF)])
    learner.load(BESTMODEL_NAME_UNF)
    return learner


def finetune_lm(train_df, config, arch, args):
    # Function to fine-tune the pre-trained language model
    if args.lang != 'en':
        tok = SentencePieceTokenizer(lang=args.lang, sp_model=SP_MODEL_LOCAL_PATH)
    else:
        tok = None
    blocks = TextBlock.from_df(args.text_col,
                               is_lm=True,
                               tok=tok
                               )

    data_block = DataBlock(blocks=blocks,
                           get_x=ColReader("text"),
                           splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE))

    lm_dataloaders =(data_block).dataloaders(train_df,
                                             bs=args.batch_size,
                                             backwards=args.bw)

    if args.lang =='en':
        pretrained_filenames = None
    else:
        pretrained_filenames = [WEIGHTS_PRETRAINED_FILE, VOCAB_PRETRAINED_FILE]

    learner_lm = language_model_learner(lm_dataloaders,
                                        arch,
                                        config=config,
                                        path=LM_MODEL_PATH,
                                        pretrained=True,
                                        pretrained_fnames=pretrained_filenames).to_fp32()

    lr = find_best_lr(learner_lm)

    learner_lm = fit_with_gradual_unfreezing(learner_lm, args.epochs, lr, args)
    learner_lm.save_encoder(ENCODER_FILE_NAME)
    return lm_dataloaders


def train_classifier(train_df, lm_dls, config, arch, args, label_list):
    # Train the classifier using the previously fine-tuned LM
    if len(label_list) > 1:
        block_category = MultiCategoryBlock()
        label_delim = args.label_delim
    else:
        block_category = CategoryBlock()
        label_delim = None

    blocks = (TextBlock.from_df(args.text_col,
                                is_lm=False,
                                seq_len=lm_dls.seq_len,
                                vocab=lm_dls.vocab,
                                tok=lm_dls.tok),
              block_category)

    clf_datablock = DataBlock(blocks=blocks,
                              get_x=ColReader("text"),
                              get_y=ColReader(LABEL_COL_NAME, label_delim=label_delim),
                              splitter=RandomSplitter(valid_pct=VAL_SIZE, seed=RANDOM_STATE)
                              )

    clf_dataloaders = clf_datablock.dataloaders(train_df,
                                                bs=args.batch_size)

    config_cls = update_classif_config(config)

    learner_clf = text_classifier_learner(clf_dataloaders,
                                          arch,
                                          path=clf_dataloaders.path,
                                          drop_mult=args.drop_mult,
                                          config=config_cls,
                                          pretrained=False).to_fp32()
    learner_clf.load_encoder(ENCODER_FILE_NAME)

    lr = find_best_lr(learner_clf)
    learner_clf = fit_with_gradual_unfreezing(learner_clf, args.epochs, lr, args)
    learner_clf.export(args.model_filename)
    return learner_clf, args.model_filename


def assess_classifier_performances(learner, test_df, args, label_list):
    # Assess model performances for each label (Accuracy, Precision and Recall)
    test_df = test_df.rename(columns={args.text_col: "text"})
    test_dataloader = learner.dls.test_dl(test_df)
    prediction_result, _ = learner.get_preds(dl=test_dataloader)
    classes = learner.dls.vocab[1]
    test_df.loc[:, PREDICTION_COL_NAME] = [
        args.label_delim.join(classes[tensor > args.prediction_threshold]) for tensor in prediction_result]
    test_df.loc[:, PREDICTION_COL_NAME] = test_df.loc[:, PREDICTION_COL_NAME].apply(
        lambda x: args.other_label if x=="" else x)
    label_scores = dict()
    for label in label_list:
        test_df.loc[:, f"{label}_predicted"] = test_df.loc[:, PREDICTION_COL_NAME].apply(lambda x: float(label in x))
        label_scores[label] = {"Accuracy": accuracy_score(test_df[label], test_df[f"{label}_predicted"]),
                               "Precision": precision_score(test_df[label], test_df[f"{label}_predicted"]),
                               "Recall": recall_score(test_df[label], test_df[f"{label}_predicted"])}
    print(label_scores)
    with open(args.score_filename, 'w') as filepath:
        json.dump(label_scores, filepath)
    return args.score_filename



def train_fastai_model(args):
    # Ensure CUDA is enabled
    print(torch.cuda.is_available())

    # Download necessary files (for english language, only the labelled dataset is necessary)
    download_file_from_gcs(args.bucket_name,
                           DATASET_GCS_PATH.format(args.lang, args.dataset_filename),
                           DATASET_LOCAL_PATH.format(args.dataset_filename))
    if args.lang != 'en':
        if args.bw:
            lm_type = LM_BACKWARD_FOLDER
        else:
            lm_type = LM_FORWARD_FOLDER
        download_file_from_gcs(args.bucket_name, VOCAB_GCS_PATH.format(args.lang, lm_type), VOCAB_LOCAL_PATH)
        download_file_from_gcs(args.bucket_name, CONFIG_GCS_PATH.format(args.lang, lm_type), CONFIG_LOCAL_PATH)
        download_file_from_gcs(args.bucket_name, WEIGHTS_GCS_PATH.format(args.lang, lm_type), WEIGHTS_LOCAL_PATH)
        download_file_from_gcs(args.bucket_name, SP_MODEL_GCS_PATH.format(args.lang), SP_MODEL_LOCAL_PATH)

    label_list = args.label_list.split(args.label_delim)

    # Format dataframe for training
    df = pd.read_csv(DATASET_LOCAL_PATH.format(args.dataset_filename))
    df.loc[:, LABEL_COL_NAME] = df.apply(_format_column_multilabels, args=(label_list, args.label_delim, args.other_label), axis=1)
    train_df, test_df = train_test_split(df.dropna(), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Open json config file
    config, arch = open_config_file(args.lang, CONFIG_LOCAL_PATH)

    # Fine-tune language model
    lm_dls = finetune_lm(train_df, config, arch, args)

    # Create and train classifier
    learner_clf, model_file_name = train_classifier(train_df, lm_dls, config, arch, args, label_list)

    # Assess model performances for each label
    label_scores_file_name = assess_classifier_performances(learner_clf,
                                                            test_df,
                                                            args,
                                                            label_list)

    return model_file_name, label_scores_file_name
