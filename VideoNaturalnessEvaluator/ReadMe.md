# Video Naturalness Evaluator
This directory contains the official code for the video naturalness evaluation model presented in the paper "[A Naturalness Evaluation Database for Video Prediction Models](https://arxiv.org/abs/2005.00356)".

## How to use:
### Training:
1. Copy the videos to Data/Predicted_Videos directory
2. Copy the MOS.csv file to Data/MOS.csv
3. Run `FeatureExtractor.py` file to extract the features from the videos.
4. Run `demo1()` method in `Train.py` file to train model. The trained model will be saved in `Trained_Models`.
5. Additionally, `demo2()` method in `Train.py` can be used to evaluate the model on 100 splits and compute median scores of PLCC, SROCC and RMSE.

### Testing:
1. To compute the naturalness score of a single video, use `demo1()` method in `Test.py`, by specifying the path to the video.
2. To compute the naturalness scores of multiple videos, place all the videos in a single directory and use the method `demo2()` in `Test.py`.
