# Shree KRISHNAya Namaha
# Runs the model and predicts quality scores
# Author: Nagabhushan S N
# Last Modified: 14-12-2021

import datetime
import json
import time
import traceback
from pathlib import Path
from typing import Union, Tuple, List, Optional

import numpy
import pandas
import skvideo.io

from feature_extractors.FeatureExtractor import PvqaFeaturesComputer
from models.ModelFactory import get_model


def get_feature_extractor(backbone_network: str, root_dirpath: Path):
    if backbone_network in ['ResNet50', 'VGG19', 'InceptionV3']:
        feature_extractor = PvqaFeaturesComputer(backbone_network=backbone_network).compute_pvqa_features
    elif backbone_network in ['C3D']:
        from feature_extractors.C3D_FeatureExtractor import C3D

        model_path = root_dirpath / 'Trained_Models/C3D/sports_1M.json'
        weights_path = root_dirpath / 'Trained_Models/C3D/sports1M_weights.h5'
        feature_extractor = C3D(model_path, weights_path).get_features
    else:
        raise RuntimeError(f'Unknown backbone network: {backbone_network}')
    return feature_extractor


def select_features(features: Union[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray], feature_names: str):
    if type(features) == numpy.ndarray:
        features = (features, )

    selected_features = []
    if 'SSA' in feature_names:
        selected_features.append(features[0])
    if 'MCS' in feature_names:
        selected_features.append(features[1])
    if 'RFD' in feature_names:
        selected_features.append(features[2])
    selected_features = numpy.concatenate(selected_features, axis=0)
    return selected_features


def load_features(root_dirpath: Path, database_name: str, backbone_network: str, video_name: str,
                  features_paths: Optional[List[Path]] = None):
    all_features = []
    if features_paths is None:
        features_paths = [
            root_dirpath / f'Data/{database_name}/Features/{backbone_network}/SSA/{video_name}.npy',
            root_dirpath / f'Data/{database_name}/Features/{backbone_network}/MCS/{video_name}.npy',
            root_dirpath / f'Data/{database_name}/Features/{backbone_network}/RFD/{video_name}.npy',
        ]
    for feature_path in features_paths:
        if feature_path.exists():
            features = numpy.load(feature_path.as_posix())
            all_features.append(features)
        else:
            break
    return all_features


def demo1():
    """
    Computes Quality Score for a single video
    :return:
    """
    root_dirpath = Path('../')
    model_dirpath = root_dirpath / 'Trained_Models/PVQA/MCS_RFD_PCA_LR_ResNet50'
    video_path = root_dirpath / 'Data/PVQA/Predicted_Videos/UCF_019.mp4'

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath

    features_computer = get_feature_extractor(train_configs['backbone_network'], root_dirpath)
    model = get_model(train_configs)
    model.load_model()

    video = skvideo.io.vread(video_path.as_posix())
    all_features = features_computer(video)
    selected_features = select_features(all_features, train_configs['features'])
    score = model.predict(selected_features).squeeze()
    print(f'Predicted Quality Score: {score:0.04f}')
    return


def demo2():
    """
    Computes quality for all the videos in a directory
    """
    root_dirpath = Path('../')
    model_dirpath = root_dirpath / 'Trained_Models/PVQA/MCS_RFD_PCA_LR_ResNet50'
    videos_dirpath = root_dirpath / 'Data/PVQA/Predicted_Videos'
    output_path = root_dirpath / 'Runs/Test01/Scores.csv'

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath

    features_computer = get_feature_extractor(train_configs['backbone_network'], root_dirpath)
    model = get_model(train_configs)
    model.load_model()

    pred_scores = []
    for video_path in sorted(videos_dirpath.iterdir()):
        video_name = video_path.stem
        start_time1 = time.time()
        video = skvideo.io.vread(video_path.as_posix())
        all_features = features_computer(video)
        selected_features = select_features(all_features, train_configs['features'])
        score = model.predict(selected_features)
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
    Computes Quality Score for a single video, for which features have been pre-computed
    :return:
    """
    root_dirpath = Path('../')
    model_dirpath = root_dirpath / 'Trained_Models/PVQA/SSA_PCA_LR_C3D'
    database_name = 'PVQA'
    video_name = 'UCF_019'

    configs_path = model_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'r') as configs_file:
        train_configs = json.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath

    model = get_model(train_configs)
    model.load_model()

    all_features = load_features(root_dirpath, database_name, train_configs['backbone_network'], video_name)
    selected_features = select_features(all_features, train_configs['features'])
    score = model.predict(selected_features).squeeze()
    print(f'Predicted Quality Score: {score:0.04f}')
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
