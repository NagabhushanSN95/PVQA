# Shree KRISHNAya Namaha
# Trains Video Naturalness Evaluator and saves the trained model
# Features must be extracted before running this file
# Author: Nagabhushan S N
# Last Modified: 05-05-2020

import datetime
import time
import traceback
from typing import Optional

import numpy
import pandas
import scipy.stats
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import Input, backend, Model
from tensorflow.keras.layers import Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from utils import CommonUtils


class VineTrainer:
    """
    Defines train, test functions
    """
    model: Optional[Model]
    num_epochs = 200

    def __init__(self) -> None:
        self.model = None

    def train(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, train_videos: numpy.ndarray):
        num_samples = train_videos.size
        train_indices = subjective_data.loc[subjective_data['Video Name'].isin(train_videos)].index.to_list()
        train_features = features[train_indices]
        train_mos = subjective_data['MOS'].to_numpy()[train_indices]
        self.model.fit(x=train_features, y=train_mos, batch_size=num_samples, epochs=self.num_epochs, verbose=1)

    def predict(self, features: numpy.ndarray, subjective_data: pandas.DataFrame,
                test_videos: numpy.ndarray) -> pandas.DataFrame:
        test_indices = subjective_data.loc[subjective_data['Video Name'].isin(test_videos)].index.to_list()
        test_features = features[test_indices]
        test_mos = subjective_data['MOS'].to_numpy()[test_indices]
        predicted_scores = self.model.predict(x=test_features)
        data = numpy.stack([test_videos, test_mos, predicted_scores.squeeze()], axis=1)
        data = pandas.DataFrame(data, columns=['Video Name', 'MOS', 'Predicted Score'])
        return data

    def test(self, features: numpy.ndarray, subjective_data: pandas.DataFrame, test_videos: numpy.ndarray):
        predicted_data = self.predict(features, subjective_data, test_videos)
        predicted_scores = predicted_data['Predicted Score'].to_numpy().astype('float')
        subjective_scores = predicted_data['MOS'].to_numpy().astype('float')
        plcc = numpy.round(scipy.stats.pearsonr(predicted_scores, subjective_scores)[0], 4)
        srocc = numpy.round(scipy.stats.spearmanr(predicted_scores, subjective_scores)[0], 4)
        rmse = numpy.round(numpy.sqrt(numpy.mean(numpy.square(predicted_scores - subjective_scores))), 4)
        return predicted_data, (plcc, srocc, rmse)

    def save_model(self, model_save_path: Path):
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_save_path.as_posix())
        return

    @staticmethod
    def load_model(model_path: Path):
        vine_model = VineTrainer()
        vine_model.model = load_model(model_path.as_posix())
        return vine_model


class VineModel(VineTrainer):
    """
    Defines the neural network model
    """

    def __init__(self) -> None:
        super().__init__()

        input_features = Input(shape=(40, 2048), name='input_features')
        mcs_features = Lambda(lambda tensors: tensors[:, 4:20])(input_features)
        rfd_features = Lambda(lambda tensors: tensors[:, 21:40])(input_features)

        # MCS processing layer
        m1f = Flatten()(mcs_features)
        m1_out = Dense(units=50, activation='relu')(m1f)

        # RFD processing layer
        m2f = Flatten()(rfd_features)
        m2_out = Dense(units=50, activation='relu')(m2f)

        merged_data = Lambda(lambda tensors: tf.concat(tensors, axis=1))([m1_out, m2_out])
        prediction = Dense(units=1, name='naturalness_output')(merged_data)
        self.model = Model(inputs=input_features, outputs=prediction)
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        self.model.summary(print_fn=print)
        plot_filepath = Path(f'../Model_Architecture/{self.__class__.__name__}.png')
        if not plot_filepath.exists():
            plot_filepath.parent.mkdir(parents=True, exist_ok=True)
            keras.utils.plot_model(self.model, plot_filepath.as_posix(), show_shapes=True)
        return


def demo1():
    """
    Trains the model and saves it
    :return:
    """
    features_dirpath = Path('../Data/Naturalness_Features')
    mos_filepath = Path('../../Data/MOS.csv')
    model_save_path = Path('../Trained_Models/VINE.h5')

    CommonUtils.init_seeds(1)
    subjective_data = pandas.read_csv(mos_filepath)
    features, scores = CommonUtils.collect_features(features_dirpath, subjective_data)
    train_videos = subjective_data['Video Name'].to_numpy()

    vine_model = VineModel()
    vine_model.train(features, subjective_data, train_videos)
    vine_model.save_model(model_save_path)
    return


def demo2():
    features_dirpath = Path('../Data/Naturalness_Features')
    mos_filepath = Path('../../Data/MOS.csv')
    tts_filepath = Path('../../Data/TrainTestSplit.csv')
    output_filepath = Path('../Runs/SplitsRun/Run01/Scores.csv')
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

        vine_model = VineModel()
        vine_model.train(features, subjective_data, train_videos)
        train_scores, train_metrics = vine_model.test(features, subjective_data, train_videos)
        test_scores, test_metrics = vine_model.test(features, subjective_data, test_videos)
        print(f'{i + 1:03}: Train PLCC: {train_metrics[0]}; Test PLCC: {test_metrics[0]}')
        print(f'{i + 1:03}: Train SROCC: {train_metrics[1]}; Test SROCC: {test_metrics[1]}')
        print(f'{i + 1:03}: Train RMSE: {train_metrics[2]}; Test RMSE: {test_metrics[2]}')
        plcc_values.append(test_metrics[0])
        srocc_values.append(test_metrics[1])
        rmse_values.append(test_metrics[2])

        all_scores = train_scores.append(test_scores).reset_index(drop=True)[['Video Name', 'Predicted Score']]
        all_scores.columns = ['Video Name', f'Split{i + 1:03}']
        combined_data = combined_data.merge(all_scores, on='Video Name')
        backend.clear_session()

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
