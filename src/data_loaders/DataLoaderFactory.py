# Shree KRISHNAya Namaha
# A factory method to return the dataloader
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

from data_loaders.PVQA import PvqaDataLoader

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_data_loader(configs: dict):
    name = configs['data_loader_name']
    if name == 'PVQA':
        model = PvqaDataLoader(configs)
    else:
        raise RuntimeError(f'Unknown data loader: {name}')
    return model
