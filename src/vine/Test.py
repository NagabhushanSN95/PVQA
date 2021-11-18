# Shree KRISHNAya Namaha
# Runs the model and predicts naturalness scores
# Author: Nagabhushan S N
# Last Modified: 14-03-2021

import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skvideo.io

import FeatureExtractor
from Train import VineModel


def demo1():
    """
    Computes Naturalness Score for a single video
    :return:
    """
    model_dirpath = Path('../../Trained_Models/VINE_ResNet50')
    video_path = Path('../../Data/Predicted_Videos/UCF_019.mp4')

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    features_computer = FeatureExtractor.VineFeaturesComputer(backbone_network=train_configs['backbone_network'])
    vine_model = VineModel.load_model(model_dirpath)

    video = skvideo.io.vread(video_path.as_posix())
    all_features = features_computer.compute_vine_features(video)
    score = vine_model.predict(all_features).squeeze()
    print(f'Predicted Naturalness Score: {score:0.04f}')
    return


def demo2():
    model_dirpath = Path('../../Trained_Models/VINE_ResNet50')
    videos_dirpath = Path('../../Data/Predicted_Videos')
    output_path = Path('../Runs/Test01/Scores.csv')

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    features_computer = FeatureExtractor.VineFeaturesComputer(backbone_network=train_configs['backbone_network'])
    vine_model = VineModel.load_model(model_dirpath)

    pred_scores = []
    for video_path in sorted(videos_dirpath.iterdir()):
        video_name = video_path.stem
        start_time1 = time.time()
        video = skvideo.io.vread(video_path.as_posix())
        vine_features = features_computer.compute_vine_features(video)
        score = vine_model.predict(vine_features)
        end_time1 = time.time()
        time_taken = end_time1 - start_time1
        print(f'{video_name}: {score:0.04f}; Time Taken: {time_taken}')
        pred_scores.append([video_name, score, time_taken])
    pred_data = pandas.DataFrame(pred_scores, columns=['Video Name', 'Predicted Score', 'Execution Time'])
    avg_score = numpy.mean(pred_data['Predicted Score'])
    avg_time = numpy.mean(pred_data['Execution Time'])
    print(f'Average Naturalness Score: {avg_score:0.04f}; Average Execution Time: {avg_time}')

    output_path.parent.mkdir(parents=True, exist_ok=False)
    pred_data.to_csv(output_path, index=False)
    return


def main():
    demo2()
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
