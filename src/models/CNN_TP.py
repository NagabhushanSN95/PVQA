# Shree KRISHNAya Namaha
# Variable Length PVQA. Uses 1d conv and adaptive pooling along time dimension.
# Features must be extracted before running this file
# Author: Nagabhushan S N
# Last Modified: 14-12-2021

from pathlib import Path

import joblib
import numpy
import pandas
import scipy.stats
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.QA import QaModel


class CnnTpModel(QaModel):
    """
    Defines train, test functions
    """

    def __init__(self, configs: dict) -> None:
        super().__init__()

        self.configs = configs
        self.features_name = configs['features']
        self.num_frames = configs['num_frames']
        self.num_time_splits = configs['num_time_splits']
        self.kernel_size = configs['kernel_size']
        self.num_filters = configs['num_filters']
        self.num_epochs = configs['num_epochs']
        self.num_frames = self.num_frames - self.kernel_size + 1

        self.model = self.get_model()
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')
        
    def get_model(self):
        if self.features_name == 'SSA':
            model = self.get_ssa_model()
        elif self.features_name == 'MCS':
            model = self.get_mcs_model()
        elif self.features_name == 'RFD':
            model = self.get_rfd_model()
        elif self.features_name == 'SSA_RFD':
            model = self.get_ssa_rfd_model()
        elif self.features_name == 'MCS_RFD':
            model = self.get_mcs_rfd_model()
        else:
            raise RuntimeError(f'Unknown model: {self.features_name}')
        return model
    
    def get_ssa_model(self):
        input_features = Input(shape=(20, 2048), name='input_features')
        ssa_features = Lambda(lambda tensors: tensors[:, :20])(input_features)

        m1_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(ssa_features)
        pooled_features = self.get_pooled_features(m1_out)

        reshaped_features = Reshape(target_shape=(self.num_filters * self.num_time_splits, ))(pooled_features)
        prediction = Dense(units=1, name='quality_output')(reshaped_features)
        model = Model(inputs=[input_features], outputs=prediction)
        return model
    
    def get_mcs_model(self):
        input_features = Input(shape=(20, 2048), name='input_features')
        mcs_features = Lambda(lambda tensors: tensors[:, 4:20])(input_features)

        m1_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(mcs_features)
        pooled_features = self.get_pooled_features(m1_out)

        reshaped_features = Reshape(target_shape=(self.num_filters * self.num_time_splits, ))(pooled_features)
        prediction = Dense(units=1, name='quality_output')(reshaped_features)
        model = Model(inputs=[input_features], outputs=prediction)
        return model
    
    def get_rfd_model(self):
        input_features = Input(shape=(20, 2048), name='input_features')
        rfd_features = Lambda(lambda tensors: tensors[:, 1:20])(input_features)

        m1_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(rfd_features)
        pooled_features = self.get_pooled_features(m1_out)

        reshaped_features = Reshape(target_shape=(self.num_filters * self.num_time_splits, ))(pooled_features)
        prediction = Dense(units=1, name='quality_output')(reshaped_features)
        model = Model(inputs=[input_features], outputs=prediction)
        return model
    
    def get_ssa_rfd_model(self):
        input_features = Input(shape=(40, 2048), name='input_features')
        ssa_features = Lambda(lambda tensors: tensors[:, :20])(input_features)
        rfd_features = Lambda(lambda tensors: tensors[:, 21:40])(input_features)

        m1_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(ssa_features)
        m2_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(rfd_features)
        merged_features = Lambda(lambda tensors: backend.concatenate(tensors, axis=2))([m1_out, m2_out])
        pooled_features = self.get_pooled_features(merged_features)

        reshaped_features = Reshape(target_shape=(2 * self.num_filters * self.num_time_splits, ))(pooled_features)
        prediction = Dense(units=1, name='quality_output')(reshaped_features)
        model = Model(inputs=[input_features], outputs=prediction)
        return model
    
    def get_mcs_rfd_model(self):
        input_features = Input(shape=(40, 2048), name='input_features')
        mcs_features = Lambda(lambda tensors: tensors[:, :20])(input_features)
        rfd_features = Lambda(lambda tensors: tensors[:, 20:40])(input_features)

        m1_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(mcs_features)
        m2_out = Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')(rfd_features)
        merged_features = Lambda(lambda tensors: backend.concatenate(tensors, axis=2))([m1_out, m2_out])
        pooled_features = self.get_pooled_features(merged_features)

        reshaped_features = Reshape(target_shape=(2 * self.num_filters * self.num_time_splits, ))(pooled_features)
        prediction = Dense(units=1, name='quality_output')(reshaped_features)
        model = Model(inputs=[input_features], outputs=prediction)
        return model
    
    def get_pooled_features(self, features):
        pooled_features = []
        for i in range(self.num_time_splits):
            t1 = int(round(i * self.num_frames / self.num_time_splits))
            t2 = int(round((i + 1) * self.num_frames / self.num_time_splits))
            pooled_feature_i = Lambda(lambda tensor: backend.mean(tensor[:, t1:t2], axis=1, keepdims=True))(features)
            pooled_features.append(pooled_feature_i)
        pooled_features = Lambda(lambda tensors: backend.concatenate(tensors, axis=1))(pooled_features)
        return pooled_features

    def train(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, train_videos: numpy.ndarray):
        train_indices = subjective_data.loc[subjective_data['Video Name'].isin(train_videos)].index.to_list()
        num_indices = len(train_indices)
        train_features = features[train_indices]
        train_mos = subjective_data['MOS'].to_numpy()[train_indices]
        self.model.fit(x=train_features, y=train_mos, batch_size=num_indices, epochs=self.num_epochs, verbose=1)
        return

    def predict(self, features: numpy.ndarray):
        predicted_score = self.model.predict(x=features[None])[0]
        return predicted_score

    def predict_all(self, features: numpy.ndarray, subjective_data: pandas.DataFrame,
                    test_videos: numpy.ndarray) -> pandas.DataFrame:
        test_indices = subjective_data.loc[subjective_data['Video Name'].isin(test_videos)].index.to_list()
        num_indices = len(test_indices)
        test_features = features[test_indices]
        test_mos = subjective_data['MOS'].to_numpy()[test_indices]
        predicted_scores = self.model.predict(x=test_features)
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

    def save_model(self, model_dirpath: Path):
        model_path = model_dirpath / 'CNN_TP.h5'
        self.model.save(model_path)
        return

    def load_model(self):
        backbone_network = self.configs['backbone_network']
        model_name = self.configs['model_name']
        features = self.configs['features']
        model_dirpath = self.configs['root_dirpath'] / self.configs['model_save_dirpath'] / f'{features}_{model_name}_{backbone_network}'
        model_path = model_dirpath / 'CNN_TP.h5'
        self.model = keras.models.load_model(model_path)
        return
