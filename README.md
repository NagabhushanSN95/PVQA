# PVQA - Predicted Videos Quality Assessment
Official Code Release of the paper "[Understanding the Understanding the Perceived Quality of Video Predictions](https://arxiv.org/abs/2005.00356)".
The database can be downloaded from the [project webpage](https://nagabhushansn95.github.io/publications/2020/pvqa.html).

## How to use:
### Training:
1. Copy the videos to Data/Predicted_Videos directory
2. Copy the MOS.csv file to Data/MOS.csv
3. Run `FeatureExtractor.py` file to extract the features from the videos.
4. Run `demo1()` method in `Train.py` file to train model. The trained model will be saved in `Trained_Models`.
5. Additionally, `demo2()` method in `Train.py` can be used to evaluate the model on 100 splits and compute median scores of PLCC, SROCC and RMSE.

### Testing:
1. To compute the quality score of a single video, use `demo1()` method in `Test.py`, by specifying the path to the video.
2. To compute the quality scores of multiple videos, place all the videos in a single directory and use the method `demo2()` in `Test.py`.

## License
Copyright 2020 Nagabhushan Somraj

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
    journal = {arXiv e-prints},
    eid = {arXiv:2005.00356},
    pages = {arXiv:2005.00356},
    archivePrefix = {arXiv},
    eprint = {2005.00356},
    year = {2020}
}
```
