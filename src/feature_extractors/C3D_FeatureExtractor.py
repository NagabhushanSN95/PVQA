# Shree KRISHNAya Namaha
# C3D Conv Net
# https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
# Author: Nagabhushan S N
# Last Modified: 14/03/2021

import datetime
import os
import time
import traceback

from tqdm import tqdm

os.environ['KERAS_BACKEND'] = 'theano'

from pathlib import Path
import numpy
import skimage.transform
import skvideo.io
from keras import backend
from keras.models import model_from_json


class C3D:
    def __init__(self, model_path: Path, weights_path: Path):
        self.model = model_from_json(open(model_path.as_posix(), 'r').read())
        print(self.model.summary())
        self.model.load_weights(weights_path.as_posix())
        self.model.compile(loss='mean_squared_error', optimizer='sgd')
        self.feature_extractor = backend.function([self.model.layers[0].input],
                                                  [self.model.layers[11].get_output(train=False)])

    def get_features(self, video: numpy.ndarray):
        frames = []
        for frame in video:
            resized_frame = skimage.transform.resize(frame, (112, 112))
            frames.append(resized_frame)
        resized_video = numpy.stack(frames)
        reshaped_video = numpy.moveaxis(resized_video[None, ...], [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])
        features = self.feature_extractor([reshaped_video])[0]
        features = numpy.moveaxis(features[0], [0, 1, 2, 3], [3, 0, 1, 2])
        return features


def save_c3d_features(videos_dirpath: Path, model_path: Path, weights_path: Path, output_dirpath: Path):
    """
    C3D features from last conv layer; Dimension: 2x7x7x512
    """
    output_dirpath.mkdir(parents=True, exist_ok=False)
    c3d_model = C3D(model_path, weights_path)
    for video_path in tqdm(sorted(videos_dirpath.rglob('*.mp4'))):
        video = skvideo.io.vread(video_path.as_posix())
        video_feature_map = c3d_model.get_features(video)
        output_path = output_dirpath / f'{video_path.stem}.npy'
        numpy.save(output_path.as_posix(), video_feature_map)
    return


def demo1():
    root_dirpath = Path('../../')
    videos_dirpath = root_dirpath / 'Data/PVQA/Predicted_Videos'
    model_path = root_dirpath / 'Trained_Models/C3D/sports_1M.json'
    weights_path = root_dirpath / 'Trained_Models/C3D/sports1M_weights.h5'
    output_dirpath = root_dirpath / 'Data/PVQA/Features/C3D/SSA'
    save_c3d_features(videos_dirpath, model_path, weights_path, output_dirpath)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
