# PVQA - Predicted Videos Quality Assessment
Official Code Release of the paper "[Understanding the Perceived Quality of Video Predictions](https://arxiv.org/abs/2005.00356)".
The database can be downloaded from the [project webpage](https://nagabhushansn95.github.io/publications/2020/pvqa.html).

## How to use:
### Training:
1. Copy the videos to `Data/PVQA/Predicted_Videos` directory.
2. Copy the `MOS.csv` file to `Data/PVQA/MOS.csv`.
3. Run `src/feature_extractors/FeatureExtractor.py` file to extract the features from the videos.
4. Run `demo1()` method in `src/Trainer.py` file to train model. The trained model will be saved in `Trained_Models/PVQA`.
5. Additionally, `demo2()` method in `src/Trainer.py` can be used to evaluate the model on 100 splits and compute median scores of PLCC, SROCC and RMSE.
6. To train on a different database, organize the videos and MOS similarly. Write a new data-loader for the new database (similar to `src/data_loaders/PVQA.py`) and change the training configs to use the new data-loader.

### Pretrained Models:
Our Model and Baseline Models pretrained on our database are available [here](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/Emlsu0iYKPZFnSi5SCjl_5EBrXTA4sANTYISdxQ3LPfOxA?e=AGA661).

### Testing:
1. To compute the quality score of a single video, use `demo1()` method in `src/Tester.py`, by specifying the path to the video.
2. To compute the quality scores of multiple videos, place all the videos in a single directory and use the method `demo2()` in `src/Tester.py`.
3. To compute the quality score of a video whose features has been already computed, use the method `demo3()` in `src/Tester.py`.
4. Since tensorflow updates the ResNet-50/VGG-19/Inception-v3 pretrained model weights with newer versions, if you use a different version of tensorflow in your setup, please train the PVQA model again instead of using the pretrained models.

If you use our PVQA model in your publication, please specify the version you are using. The current version is 1.3.1.

## License
Copyright 2020 Nagabhushan Somraj, Manoj Surya Kashi, S P Arun, Rajiv Soundararajan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this code except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Citation
If you use this code for your research, please cite our paper

```bibtex
@article{somraj2020pvqa,
    title = {Understanding the Perceived Quality of Video Predictions},
    author = {Somraj, Nagabhushan and Kashi, Manoj Surya and Arun, S. P. and Soundararajan, Rajiv},
    journal = {Signal Processing: Image Communication},
    volume = {102},
    pages = {116626},
    issn = {0923-5965},
    year = {2022},
    doi = {https://doi.org/10.1016/j.image.2021.116626}
}
```
