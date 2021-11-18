# Shree KRISHNAya Namaha
# Runs the model and predicts naturalness scores
# Author: Nagabhushan S N
# Last Modified: 05-05-2020

import datetime
import time
import traceback
from pathlib import Path

import numpy
import skvideo.io
import tensorflow as tf     # This is required for loading the pretrained model

import FeatureExtractor
from Train import VineModel


def demo1():
    """
    Computes Naturalness Score for a single video
    :return:
    """
    model_path = Path('../Trained_Models/VINE.h5')
    video_path = Path('../Data/Predicted_Videos/UCF_019.mp4')

    video = skvideo.io.vread(video_path.as_posix())
    all_features = FeatureExtractor.get_all_features(video)
    vine_model = VineModel.load_model(model_path)
    score = vine_model.model.predict(all_features[None]).squeeze()
    print(f'Predicted Naturalness Score: {score:0.04f}')
    return


def demo2():
    model_path = Path('../Trained_Models/VINE.h5')
    videos_dirpath = Path('../../Data/Predicted_Videos')

    pred_scores = []
    feature_extractor = FeatureExtractor.ResNetFeatureExtractor()
    for video_path in sorted(videos_dirpath.iterdir()):
        video_name = video_path.stem
        video = skvideo.io.vread(video_path.as_posix())
        all_features = FeatureExtractor.get_all_features(video, feature_extractor)
        vine_model = VineModel.load_model(model_path)
        score = vine_model.model.predict(all_features[None]).squeeze()
        print(f'{video_name}: {score:0.04f}')
        pred_scores.append(score)
    avg_score = numpy.mean(pred_scores)
    print(f'Average Naturalness Score: {avg_score:0.04f}')
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
