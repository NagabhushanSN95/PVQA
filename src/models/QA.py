# Shree KRISHNAya Namaha
# A parent abstract class
# Author: Nagabhushan S N
# Last Modified: 14/12/21

import abc
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


class QaModel:

    @abc.abstractmethod
    def train(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, train_video_names: numpy.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, features: numpy.ndarray) -> float:
        pass

    @abc.abstractmethod
    def predict_all(self, features: numpy.ndarray, subjective_data: pandas.DataFrame,
                    test_videos: numpy.ndarray) -> pandas.DataFrame:
        pass

    @abc.abstractmethod
    def test(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, test_videos: numpy.ndarray):
        pass

    @abc.abstractmethod
    def save_model(self, model_save_dirpath: Path):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_model(model_dirpath: Path):
        pass
