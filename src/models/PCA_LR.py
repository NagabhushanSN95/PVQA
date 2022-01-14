# Shree KRISHNAya Namaha
# Trains PVQA Models on the provided features and saves the trained model
# Features must be extracted before running this file
# Author: Nagabhushan S N
# Last Modified: 14-12-2021
import json
from pathlib import Path

import joblib
import numpy
import pandas
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from models.QA import QaModel


class PcaLrModel(QaModel):
    """
    Defines train, test functions
    """
    model: LinearRegression
    pca: PCA

    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.num_principal_components = configs['num_principal_components']
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

    def load_model(self):
        backbone_network = self.configs['backbone_network']
        model_name = self.configs['model_name']
        features = self.configs['features']
        model_dirpath = self.configs['root_dirpath'] / self.configs['model_save_dirpath'] / f'{features}_{model_name}_{backbone_network}'
        pca_save_path = model_dirpath / 'PCA.joblib'
        lr_save_path = model_dirpath / 'LinearRegression.joblib'
        self.pca = joblib.load(pca_save_path)
        self.regressor = joblib.load(lr_save_path)
        return
