# Shree KRISHNAya Namaha
# Common Utils
# Author: Nagabhushan S N
# Last Modified: 01/04/2020

import os
import random
from pathlib import Path

import numpy
import pandas
import tensorflow as tf


def init_seeds(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)
    return


def get_tts_video_names(split_filepath: Path, group_no: int) -> dict:
    """
    Returns only video names for train/test splits.
    :param split_filepath:
    :param group_no: Starts at 0
    :return:
    """
    split_data = pandas.read_csv(split_filepath)
    split_name = split_data.keys()[group_no + 1]

    groups = pandas.unique(split_data[split_name]).tolist()  # Train, Test, Validation
    split_video_names = {}
    for group in groups:
        group_video_names = split_data.loc[split_data[split_name] == group]['Video Name'].to_numpy()
        split_video_names[group] = group_video_names

    split_data = {
        'Split Name': split_name,
        'Video Names': split_video_names
    }
    return split_data


def collect_features(features_dirpath: Path, subjective_data: pandas.DataFrame):
    all_features = []
    all_scores = []
    for i, row in subjective_data.iterrows():
        video_name = row['Video Name']
        features_path = features_dirpath / f'{video_name}.npy'
        features = numpy.load(features_path.as_posix())
        all_features.append(features)
        score = row['MOS']
        all_scores.append(score)
    all_features = numpy.stack(all_features, axis=0)
    all_scores = numpy.array(all_scores)
    return all_features, all_scores
