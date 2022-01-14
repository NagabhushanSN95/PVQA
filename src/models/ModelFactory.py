# Shree KRISHNAya Namaha
# A factory method to return the model
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

from models.PCA_LR import PcaLrModel
from models.CNN_TP import CnnTpModel

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_model(configs: dict):
    name = configs['model_name']
    if name == 'PCA_LR':
        model = PcaLrModel(configs)
    elif name == 'CNN_TP':
        model = CnnTpModel(configs)
    else:
        raise RuntimeError(f'Unknown model: {name}')
    return model
