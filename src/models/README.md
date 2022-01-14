# Models

### PCA_LR (Principal Component Analysis + Linear Regression)
Regression model presented in our paper.

### CNN_TP (Convolutional Neural Network + Temporal Pooling)
Extension of our model that can work on arbitrary length videos. It uses 1d CNNs along the time dimension, splits the sequence into 4 parts and temporally averages the features in each part. Finally, a dense layer is used to predict the quality score.  
Note: This model has a drop in performance by about 0.02 in SROCC and PLCC as compared to the PCA_LR model.
