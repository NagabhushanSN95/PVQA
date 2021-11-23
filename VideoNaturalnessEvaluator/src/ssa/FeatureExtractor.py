# Shree KRISHNAya Namaha
# Extracts MCS and RFD features
# Author: Nagabhushan S N
# Last Modified: 14-03-2021

import abc
import datetime
import time
import traceback
from pathlib import Path

import numpy
import skvideo.io
from tqdm import tqdm


class DeepFeaturesExtractor:
    @abc.abstractmethod
    def compute_features(self, video: numpy.ndarray) -> numpy.ndarray:
        pass


class ResNetFeatureExtractor(DeepFeaturesExtractor):
    def __init__(self) -> None:
        from tensorflow.keras.applications.resnet50 import ResNet50
        self.model = ResNet50(include_top=False)

    def compute_features(self, video: numpy.ndarray) -> numpy.ndarray:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        preprocessed_video = preprocess_input(video)
        features = self.model.predict(preprocessed_video)
        return features


class Vgg19FeatureExtractor(DeepFeaturesExtractor):
    def __init__(self) -> None:
        from tensorflow.keras.applications.vgg19 import VGG19
        self.model = VGG19(include_top=False)

    def compute_features(self, video: numpy.ndarray) -> numpy.ndarray:
        from tensorflow.keras.applications.vgg19 import preprocess_input
        preprocessed_video = preprocess_input(video)
        features = self.model.predict(preprocessed_video)
        return features


class InceptionV3FeatureExtractor(DeepFeaturesExtractor):
    def __init__(self) -> None:
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        self.model = InceptionV3(include_top=False)

    def compute_features(self, video: numpy.ndarray) -> numpy.ndarray:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        preprocessed_video = preprocess_input(video)
        features = self.model.predict(preprocessed_video)
        return features


class SsaFeaturesComputer:
    def __init__(self, backbone_network: str = 'ResNet50'):
        self.deep_features_extractor = self.get_deep_features_extractor(backbone_network)
        return

    @staticmethod
    def get_deep_features_extractor(network: str) -> DeepFeaturesExtractor:
        if network == 'ResNet50':
            deep_features_extractor = ResNetFeatureExtractor()
        elif network == 'VGG19':
            deep_features_extractor = Vgg19FeatureExtractor()
        elif network == 'InceptionV3':
            deep_features_extractor = InceptionV3FeatureExtractor()
        else:
            raise RuntimeError(f'Unknown backbone network: {network}')
        return deep_features_extractor

    def compute_ssa_features(self, video: numpy.ndarray) -> numpy.ndarray:
        video_features = self.deep_features_extractor.compute_features(video)
        ssa_features = numpy.mean(video_features, axis=(1, 2))
        return ssa_features


def extract_features(videos_dirpath: Path, backbone_network: str, output_dirpath: Path):
    features_computer = SsaFeaturesComputer(backbone_network)
    output_dirpath = output_dirpath / backbone_network
    output_dirpath.mkdir(parents=True, exist_ok=False)
    print('Computing SSA features')
    for video_path in tqdm(sorted(videos_dirpath.iterdir())):
        video_name = video_path.stem
        video = skvideo.io.vread(video_path.as_posix())
        all_features = features_computer.compute_ssa_features(video)

        output_filepath = output_dirpath / f'{video_name}.npy'
        numpy.save(output_filepath.as_posix(), all_features)
    print('Features computation complete')
    return


def demo1():
    root_dirpath = Path('../../')
    videos_dirpath = root_dirpath / 'Data/Predicted_Videos'
    backbone_network = 'ResNet50'
    output_dirpath = root_dirpath / 'Data/SSA_Features'
    extract_features(videos_dirpath, backbone_network, output_dirpath)
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
