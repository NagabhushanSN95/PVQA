# Shree KRISHNAya Namaha
# Trains Video Naturalness Evaluator and saves the trained model
# Features must be extracted before running this file
# Author: Nagabhushan S N
# Last Modified: 05-05-2020

import datetime
import time
import traceback

import joblib
import numpy
import pandas
import scipy.stats
import simplejson as simplejson
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from utils import CommonUtils


class SsaModel:
    """
    Defines train, test functions
    """
    model: LinearRegression
    pca: PCA

    def __init__(self, num_components=240) -> None:
        self.num_principal_components = num_components
        self.regressor = LinearRegression()
        self.pca = PCA(n_components=self.num_principal_components)
        return

    def train(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, train_videos: numpy.ndarray):
        train_indices = subjective_data.loc[subjective_data['Video Name'].isin(train_videos)].index.to_list()
        num_indices = len(train_indices)
        train_features = features[train_indices].reshape(num_indices, -1)
        reduced_features = self.pca.fit_transform(train_features)
        train_mos = subjective_data['MOS'].to_numpy()[train_indices]
        self.regressor.fit(X=reduced_features, y=train_mos)
        return

    def predict(self, features: numpy.ndarray):
        batch_features = numpy.reshape(features, (1, -1))
        reduced_features = self.pca.transform(X=batch_features)
        predicted_score = self.regressor.predict(X=reduced_features)[0]
        return predicted_score

    def predict_all(self, features: numpy.ndarray, subjective_data: pandas.DataFrame,
                    test_videos: numpy.ndarray) -> pandas.DataFrame:
        test_indices = subjective_data.loc[subjective_data['Video Name'].isin(test_videos)].index.to_list()
        num_indices = len(test_indices)
        test_features = features[test_indices].reshape(num_indices, -1)
        reduced_features = self.pca.transform(X=test_features)
        test_mos = subjective_data['MOS'].to_numpy()[test_indices]
        predicted_scores = self.regressor.predict(X=reduced_features)
        data = numpy.stack([test_videos, test_mos, predicted_scores.squeeze()], axis=1)
        data = pandas.DataFrame(data, columns=['Video Name', 'MOS', 'Predicted Score'])
        return data

    def test(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, test_videos: numpy.ndarray):
        predicted_data = self.predict_all(features, subjective_data, test_videos)
        predicted_scores = predicted_data['Predicted Score'].to_numpy().astype('float')
        subjective_scores = predicted_data['MOS'].to_numpy().astype('float')
        plcc = numpy.round(scipy.stats.pearsonr(predicted_scores, subjective_scores)[0], 4)
        srocc = numpy.round(scipy.stats.spearmanr(predicted_scores, subjective_scores)[0], 4)
        rmse = numpy.round(numpy.sqrt(numpy.mean(numpy.square(predicted_scores - subjective_scores))), 4)
        return predicted_data, (plcc, srocc, rmse)

    def save_model(self, model_save_dirpath: Path):
        pca_save_path = model_save_dirpath / 'PCA.joblib'
        lr_save_path = model_save_dirpath / 'LinearRegression.joblib'
        joblib.dump(self.pca, pca_save_path)
        joblib.dump(self.regressor, lr_save_path)
        return

    @staticmethod
    def load_model(model_dirpath: Path):
        ssa_model = SsaModel()
        pca_save_path = model_dirpath / 'PCA.joblib'
        lr_save_path = model_dirpath / 'LinearRegression.joblib'
        ssa_model.pca = joblib.load(pca_save_path)
        ssa_model.regressor = joblib.load(lr_save_path)
        return ssa_model


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    configs = configs.copy()
    for key in configs.keys():
        if isinstance(configs[key], Path):
            configs[key] = configs[key].as_posix()

    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def demo1():
    """
    Trains the model and saves it
    :return:
    """
    root_dirpath = Path('../../')
    configs = {
        'backbone_network': 'ResNet50',
        'root_dirpath': root_dirpath,
        'features_dirpath': root_dirpath / 'Data/SSA_Features',
        'mos_filepath': root_dirpath / 'Data/MOS.csv',
        'model_save_dirpath': root_dirpath / 'Trained_Models',
        'seed': 1,
    }
    backbone_network = configs['backbone_network']
    features_dirpath = configs['features_dirpath'] / backbone_network
    mos_filepath = configs['mos_filepath']
    model_save_dirpath = configs['model_save_dirpath'] / f'SSA_{backbone_network}'
    model_save_dirpath.mkdir(parents=True, exist_ok=False)
    seed = configs['seed']
    save_configs(model_save_dirpath, configs)

    CommonUtils.init_seeds(seed)
    subjective_data = pandas.read_csv(mos_filepath)
    features, scores = CommonUtils.collect_features(features_dirpath, subjective_data)
    train_videos = subjective_data['Video Name'].to_numpy()

    ssa_model = SsaModel()
    ssa_model.train(features, subjective_data, train_videos)
    ssa_model.save_model(model_save_dirpath)
    return


def demo2():
    features_dirpath = Path('../../Data/SSA_Features/ResNet50')
    mos_filepath = Path('../../Data/MOS.csv')
    tts_filepath = Path('../../Data/TrainTestSplit.csv')
    output_filepath = Path('../../TimingAnalysis/SplitsRun/Run04/Scores.csv')
    num_splits = 100

    if output_filepath and output_filepath.exists():
        raise RuntimeError(f'File already exists: {output_filepath.as_posix()}')

    start_time1 = time.time()
    CommonUtils.init_seeds(1)
    subjective_data = pandas.read_csv(mos_filepath)
    features, scores = CommonUtils.collect_features(features_dirpath, subjective_data)

    plcc_values, srocc_values, rmse_values = [], [], []
    combined_data = subjective_data[['Video Name', 'MOS']]

    for i in range(num_splits):
        split_data = CommonUtils.get_tts_video_names(tts_filepath, i)
        train_videos = split_data['Video Names']['Train']
        test_videos = split_data['Video Names']['Test']

        ssa_model = SsaModel()
        ssa_model.train(features, subjective_data, train_videos)
        train_scores, train_metrics = ssa_model.test(features, subjective_data, train_videos)
        test_scores, test_metrics = ssa_model.test(features, subjective_data, test_videos)
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

    if output_filepath:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        combined_data.to_csv(output_filepath, index=False)
    end_time1 = time.time()
    exec_time1 = str(datetime.timedelta(seconds=end_time1 - start_time1))
    return plcc_values, srocc_values, rmse_values, exec_time1


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
