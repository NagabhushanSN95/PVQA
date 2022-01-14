# Shree KRISHNAya Namaha
# Trains PVQA Models and saves the trained model
# Features must be extracted before running this file
# Author: Nagabhushan S N
# Last Modified: 14/01/2022

import datetime
import time
import traceback

import numpy
import pandas
import simplejson as simplejson
from pathlib import Path

from data_loaders.DataLoaderFactory import get_data_loader
from models.ModelFactory import get_model
from utils import CommonUtils


def save_configs(output_dirpath: Path, configs: dict):
    output_dirpath.mkdir(parents=True, exist_ok=True)
    configs_path = output_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def demo1():
    """
    Trains the model and saves it
    :return:
    """
    root_dirpath = Path('../')
    configs = {
        'features_dirpath': 'Data/PVQA/Features',
        'mos_filepath': 'Data/PVQA/MOS.csv',
        'tts_filepath': 'Data/PVQA/TrainTestSplit.csv',
        'model_save_dirpath': 'Trained_Models/PVQA',
        'backbone_network': 'ResNet50',
        'features': 'MCS_RFD',
        'model_name': 'PCA_LR',
        'data_loader_name': 'PVQA',
        'num_principal_components': 240,
        'seed': 1,
    }
    # configs = {
    #     'features_dirpath': 'Data/PVQA/Features',
    #     'mos_filepath': 'Data/PVQA/MOS.csv',
    #     'tts_filepath': 'Data/PVQA/TrainTestSplit.csv',
    #     'model_save_dirpath': 'Trained_Models/PVQA',
    #     'backbone_network': 'ResNet50',
    #     'features': 'MCS_RFD',
    #     'model_name': 'CNN_TP',
    #     'data_loader_name': 'PVQA',
    #     'num_frames': 20,
    #     'num_context_frames': 4,
    #     'num_time_splits': 4,
    #     'kernel_size': 5,
    #     'num_filters': 100,
    #     'num_epochs': 200,
    #     'seed': 1,
    # }

    backbone_network = configs['backbone_network']
    model_name = configs['model_name']
    features = configs['features']
    model_save_dirpath = root_dirpath / configs['model_save_dirpath'] / f'{features}_{model_name}_{backbone_network}'
    model_save_dirpath.mkdir(parents=True, exist_ok=False)
    save_configs(model_save_dirpath, configs)
    configs['root_dirpath'] = root_dirpath

    seed = configs['seed']
    CommonUtils.init_seeds(seed)

    data_loader = get_data_loader(configs)
    features, scores = data_loader.load_data()
    subjective_data = pandas.read_csv(data_loader.mos_path)
    train_videos = subjective_data['Video Name'].to_numpy()

    model = get_model(configs)
    model.train(features, subjective_data, train_videos)
    model.save_model(model_save_dirpath)
    return


def demo2():
    root_dirpath = Path('../')
    configs = {
        'features_dirpath': 'Data/PVQA/Features',
        'mos_filepath': 'Data/PVQA/MOS.csv',
        'tts_filepath': 'Data/PVQA/TrainTestSplit.csv',
        'num_splits': 100,
        'output_dirpath': 'Runs/SplitsRun/Run09',
        'backbone_network': 'C3D',
        'features': 'SSA',
        'model_name': 'PCA_LR',
        'data_loader_name': 'PVQA',
        'num_principal_components': 240,
        'seed': 1,
    }
    # configs = {
    #     'features_dirpath': 'Data/PVQA/Features',
    #     'mos_filepath': 'Data/PVQA/MOS.csv',
    #     'tts_filepath': 'Data/PVQA/TrainTestSplit.csv',
    #     'num_splits': 100,
    #     'output_dirpath': 'Runs/SplitsRun/Run04',
    #     'backbone_network': 'ResNet50',
    #     'features': 'MCS_RFD',
    #     'model_name': 'CNN_TP',
    #     'data_loader_name': 'PVQA',
    #     'num_frames': 20,
    #     'num_context_frames': 4,
    #     'num_time_splits': 4,
    #     'kernel_size': 5,
    #     'num_filters': 100,
    #     'num_epochs': 200,
    #     'seed': 1,
    # }

    tts_filepath = root_dirpath / configs['tts_filepath']
    output_dirpath = root_dirpath / configs['output_dirpath']
    scores_filepath = output_dirpath / 'Scores.csv'

    if scores_filepath and scores_filepath.exists():
        raise RuntimeError(f'File already exists: {scores_filepath.as_posix()}')

    save_configs(output_dirpath, configs)
    configs['root_dirpath'] = root_dirpath

    start_time1 = time.time()
    seed = configs['seed']
    CommonUtils.init_seeds(seed)

    data_loader = get_data_loader(configs)
    features, scores = data_loader.load_data()
    subjective_data = pandas.read_csv(data_loader.mos_path)

    plcc_values, srocc_values, rmse_values = [], [], []
    combined_data = subjective_data[['Video Name', 'MOS']]

    for i in range(configs['num_splits']):
        split_data = CommonUtils.get_tts_video_names(tts_filepath, i)
        train_videos = split_data['Video Names']['Train']
        test_videos = split_data['Video Names']['Test']

        model = get_model(configs)
        model.train(features, subjective_data, train_videos)
        train_scores, train_metrics = model.test(features, subjective_data, train_videos)
        test_scores, test_metrics = model.test(features, subjective_data, test_videos)
        print(f'{i + 1:03}: Train PLCC: {train_metrics[0]}; Test PLCC: {test_metrics[0]}')
        print(f'{i + 1:03}: Train SROCC: {train_metrics[1]}; Test SROCC: {test_metrics[1]}')
        print(f'{i + 1:03}: Train RMSE: {train_metrics[2]}; Test RMSE: {test_metrics[2]}')
        plcc_values.append(test_metrics[0])
        srocc_values.append(test_metrics[1])
        rmse_values.append(test_metrics[2])

        all_scores = train_scores.append(test_scores).reset_index(drop=True)[['Video Name', 'Predicted Score']]
        all_scores.columns = ['Video Name', f'Split{i + 1:03}']
        combined_data = combined_data.merge(all_scores, on='Video Name')

    print(plcc_values)
    print(srocc_values)
    print(rmse_values)
    num_finite_scores = min(numpy.isfinite(plcc_values).sum(), numpy.isfinite(srocc_values).sum(),
                            numpy.isfinite(rmse_values).sum())
    plcc_median = numpy.round(numpy.nanmedian(plcc_values), 4)
    plcc_std = numpy.round(numpy.nanstd(plcc_values), 4)
    srocc_median = numpy.round(numpy.nanmedian(srocc_values), 4)
    srocc_std = numpy.round(numpy.nanstd(srocc_values), 4)
    rmse_median = numpy.round(numpy.nanmedian(rmse_values), 4)
    rmse_std = numpy.round(numpy.nanstd(rmse_values), 4)
    print()
    print(f'Num Finite Scores: {num_finite_scores}')
    print(f'PLCC: Median: {plcc_median}; Std: {plcc_std}')
    print(f'SROCC: Median: {srocc_median}; Std: {srocc_std}')
    print(f'RMSE: Median: {rmse_median}; Std: {rmse_std}')

    if scores_filepath:
        scores_filepath.parent.mkdir(parents=True, exist_ok=True)
        combined_data.to_csv(scores_filepath, index=False)
    end_time1 = time.time()
    exec_time1 = str(datetime.timedelta(seconds=end_time1 - start_time1))
    return plcc_values, srocc_values, rmse_values, exec_time1


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
