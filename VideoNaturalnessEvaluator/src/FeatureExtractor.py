# Shree KRISHNAya Namaha
# Extracts MCS and RFD features
# To be run inside src/ folder
# Author: Nagabhushan S N
# Last Modified: 04-05-2020

import datetime
import time
import traceback
from pathlib import Path

import numpy
import skvideo.io
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm


class ResNetFeatureExtractor:
    def __init__(self) -> None:
        self.model = ResNet50(include_top=False)

    def get_features(self, video: numpy.ndarray) -> numpy.ndarray:
        preprocessed_video = preprocess_input(video)
        features = self.model.predict(preprocessed_video)
        return features


# ------------------------------ MCS Features ---------------------------------------- #
def spatial_cosine_sim(features1: numpy.ndarray, features2: numpy.ndarray) -> numpy.ndarray:
    # noinspection PyTypeChecker
    cosine_sim = numpy.einsum('ijkl,ijkl->il', features1, features2) / \
                 (numpy.linalg.norm(features1, axis=(1, 2)) * numpy.linalg.norm(features2, axis=(1, 2)) + 1e-15)
    return cosine_sim


def channel_cosine_sim(features1: numpy.ndarray, features2: numpy.ndarray):
    cosine_sim = numpy.einsum('ijk,ijk->ij', features1, features2) / \
                 (numpy.linalg.norm(features1, axis=2) * numpy.linalg.norm(features2, axis=2) + 1e-15)
    return cosine_sim


def compute_matched_indices(ref_feature: numpy.ndarray, test_feature: numpy.ndarray) -> numpy.ndarray:
    h, w = ref_feature.shape[:2]
    matched_indices = -1 * numpy.zeros(shape=(h, w), dtype=int)
    for h1 in range(h):
        for w1 in range(w):
            cos_sim = channel_cosine_sim(ref_feature[h1:h1 + 1, w1:w1 + 1], test_feature)
            matched_index = numpy.argmax(cos_sim)
            matched_indices[h1, w1] = matched_index
    return matched_indices


def get_motion_compensated_features(features1: numpy.ndarray, motion_indices: numpy.ndarray):
    h, w = motion_indices.shape
    motion_compensated_features = numpy.zeros(shape=features1.shape, dtype=features1.dtype)
    for i in range(h):
        for j in range(w):
            index = motion_indices[i, j]
            i1, j1 = numpy.unravel_index(index, shape=motion_indices.shape)
            motion_compensated_features[i, j] = features1[i1, j1]
    return motion_compensated_features


def matched_cosine_sim(features1: numpy.ndarray, features2: numpy.ndarray):
    """
    features2 is rearranged to match features1 and then spatial cosine similarity is computed
    """
    num_frames = max(features1.shape[0], features2.shape[0])
    if features1.shape[0] == 1:
        features1 = numpy.repeat(features1, repeats=num_frames, axis=0)
    if features2.shape[0] == 1:
        features2 = numpy.repeat(features2, repeats=num_frames, axis=0)

    matched_features = []
    for i in range(num_frames):
        matched_indices = compute_matched_indices(features1[i], features2[i])
        matched_feature = get_motion_compensated_features(features2[i], matched_indices)
        matched_features.append(matched_feature)
    matched_features = numpy.stack(matched_features)
    cosine_features = spatial_cosine_sim(features1, matched_features)
    return cosine_features


def get_mcs_features(video_features: numpy.ndarray) -> numpy.ndarray:
    mcs_features = matched_cosine_sim(video_features[3:4], video_features)
    return mcs_features


# ------------------------------ RFD Features ---------------------------------------- #
def prepend_zero_frame(video_frames: numpy.ndarray) -> numpy.ndarray:
    zero_frame = numpy.zeros(video_frames.shape[1:])[None]
    padded_frames = numpy.concatenate([zero_frame, video_frames], axis=0)
    return padded_frames


def get_rfd_video(video: numpy.ndarray) -> numpy.ndarray:
    float_video = video.astype('float')
    diff_video = float_video[1:] - float_video[:-1]
    norm_video = (diff_video - diff_video.min()) / (diff_video.max() - diff_video.min())
    rfd_video = (norm_video * 255).astype('uint8')
    return rfd_video


def get_rfd_features(diff_features: numpy.ndarray) -> numpy.ndarray:
    rfd_features = prepend_zero_frame(numpy.mean(diff_features, axis=(1, 2)))
    return rfd_features


# -------------------------- Features Aggregator ---------------------------- #
def get_all_features(video: numpy.ndarray, feature_extractor: ResNetFeatureExtractor = None):
    if not feature_extractor:
        feature_extractor = ResNetFeatureExtractor()
    rfd_video = get_rfd_video(video)

    video_features = feature_extractor.get_features(video)
    diff_features = feature_extractor.get_features(rfd_video)

    mcs_features = get_mcs_features(video_features)
    rfd_features = get_rfd_features(diff_features)
    all_features = numpy.concatenate([mcs_features, rfd_features], axis=0)
    return all_features


def extract_features(videos_dirpath: Path, output_dirpath: Path):
    output_dirpath.mkdir(parents=True, exist_ok=False)
    print('Extracting MCS and RFD features')
    feature_extractor = ResNetFeatureExtractor()
    for video_path in tqdm(sorted(videos_dirpath.iterdir())):
        video_name = video_path.stem
        video = skvideo.io.vread(video_path.as_posix())
        all_features = get_all_features(video, feature_extractor)

        output_filepath = output_dirpath / f'{video_name}.npy'
        numpy.save(output_filepath.as_posix(), all_features)
    print('Extracting features complete')
    return


def demo1():
    videos_dirpath = Path('../../Data/Predicted_Videos')
    output_dirpath = Path('../Data/Naturalness_Features')
    extract_features(videos_dirpath, output_dirpath)
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
