# Shree KRISHNAya Namaha
# Loads pre-computed features for the PVQA database
# Author: Nagabhushan S N
# Last Modified: 14/12/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class PvqaDataLoader:
    def __init__(self, configs: dict):
        self.configs = configs
        self.root_dirpath = configs['root_dirpath']
        self.features_dirpath = self.root_dirpath / 'Data/PVQA/Features'
        self.mos_path = self.root_dirpath / 'Data/PVQA/MOS.csv'
        return

    def load_data(self):
        subjective_data = pandas.read_csv(self.mos_path)
        backbone_network = self.configs['backbone_network']
        feature_names = self.configs['features'].split('_')
        features, scores = [], []
        for i, row in subjective_data.iterrows():
            video_name = row['Video Name']
            video_features = []
            for feature_name in feature_names:
                feature_path = self.features_dirpath / f'{backbone_network}/{feature_name}/{video_name}.npy'
                named_features = numpy.load(feature_path.as_posix())
                video_features.append(named_features)
            video_features = numpy.concatenate(video_features, axis=0)
            features.append(video_features)
            score = row['MOS']
            scores.append(score)
        features = numpy.stack(features, axis=0)
        scores = numpy.array(scores)
        return features, scores
