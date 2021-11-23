# Shree KRISHNAya Namaha
# Runs the SSA baseline model and predicts the quality scores
# Author: Nagabhushan S N
# Last Modified: 18-11-2021

import datetime
import json
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skvideo.io

from ssa import C3D_FeatureExtractor
from ssa.FeatureExtractor import SsaFeaturesComputer
from ssa.Train import SsaModel


def demo1():
    """
    Computes quality score for a single video
    :return:
    """
    model_dirpath = Path('../../Trained_Models/SSA_ResNet50')
    video_path = Path('../../Data/Predicted_Videos/UCF_019.mp4')

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    features_computer = SsaFeaturesComputer(backbone_network=train_configs['backbone_network'])
    ssa_model = SsaModel.load_model(model_dirpath)

    video = skvideo.io.vread(video_path.as_posix())
    all_features = features_computer.compute_ssa_features(video)
    score = ssa_model.predict(all_features).squeeze()
    print(f'Predicted Quality Score: {score:0.04f}')
    return


def demo2():
    model_dirpath = Path('../../Trained_Models/SSA_ResNet50')
    videos_dirpath = Path('../../Data/Predicted_Videos')
    output_path = Path('../../TimingAnalysis/Test04_SSA_ResNet50/Scores.csv')

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    features_computer = SsaFeaturesComputer(backbone_network=train_configs['backbone_network'])
    ssa_model = SsaModel.load_model(model_dirpath)

    pred_scores = []
    for video_path in sorted(videos_dirpath.iterdir()):
        video_name = video_path.stem
        start_time1 = time.time()
        video = skvideo.io.vread(video_path.as_posix())
        pvqa_features = features_computer.compute_ssa_features(video)
        score = ssa_model.predict(pvqa_features)
        end_time1 = time.time()
        time_taken = end_time1 - start_time1
        print(f'{video_name}: {score:0.04f}; Time Taken: {time_taken}')
        pred_scores.append([video_name, score, time_taken])
    pred_data = pandas.DataFrame(pred_scores, columns=['Video Name', 'Predicted Score', 'Execution Time'])
    avg_score = numpy.mean(pred_data['Predicted Score'])
    avg_time = numpy.mean(pred_data['Execution Time'])
    print(f'Average Quality Score: {avg_score:0.04f}; Average Execution Time: {avg_time}')

    output_path.parent.mkdir(parents=True, exist_ok=False)
    pred_data.to_csv(output_path, index=False)
    return


def demo3():
    """
    For SSA-C3D
    :return:
    """
    model_dirpath = Path('../../Trained_Models/SSA_C3D')
    videos_dirpath = Path('../../Data/Predicted_Videos')
    output_path = Path('../../TimingAnalysis/Test05_SSA_C3D/Scores.csv')

    # configs_path = model_dirpath / 'Configs.json'
    # with open(configs_path.as_posix(), 'r') as configs_file:
    #     train_configs = json.load(configs_file)
    # features_computer = SsaFeaturesComputer(backbone_network=train_configs['backbone_network'])
    c3d_arch_path = Path('../../Trained_Models/C3D/sports1M.json')
    c3d_weights_path = Path('../../Trained_Models/C3D/sports1M_weights.h5')
    features_computer = C3D_FeatureExtractor.C3D(c3d_arch_path, c3d_weights_path)
    ssa_model = SsaModel.load_model(model_dirpath)

    pred_scores = []
    for video_path in sorted(videos_dirpath.iterdir()):
        video_name = video_path.stem
        start_time1 = time.time()
        video = skvideo.io.vread(video_path.as_posix())
        pvqa_features = features_computer.get_features(video)
        score = ssa_model.predict(pvqa_features)
        end_time1 = time.time()
        time_taken = end_time1 - start_time1
        print(f'{video_name}: {score:0.04f}; Time Taken: {time_taken}')
        pred_scores.append([video_name, score, time_taken])
    pred_data = pandas.DataFrame(pred_scores, columns=['Video Name', 'Predicted Score', 'Execution Time'])
    avg_score = numpy.mean(pred_data['Predicted Score'])
    avg_time = numpy.mean(pred_data['Execution Time'])
    print(f'Average Quality Score: {avg_score:0.04f}; Average Execution Time: {avg_time}')

    output_path.parent.mkdir(parents=True, exist_ok=False)
    pred_data.to_csv(output_path, index=False)
    return


def main():
    demo3()
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