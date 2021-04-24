
Preprocessing (xr_preprocess.py):
After collecting the data from Geostationary Operational Environmental Satellite (GOES), we noticed that the magnitude of the 
flux is very small. So we first normalize the data. There are also some noise as the satellites start to degrade. That is why
we decided to replace those invalid values with the previous values. After that we used exponential filtering to assign more weights
to the recent flux. Then we used the gamma correction. 

Neural Networks:
We have explored Recurrent Neural Network(LSTM), Convolutional Neural Network(1D CNN) and a N-BEATS to predict the solar flux.
 


We have also analysed the data using Principal Component Analysis (xl_max_from_xl_pca_cnn1d_v1.py)

As the flares are rare events, there are more non-flare samples compared to the flare samples. That is why we have also used
data augmentation to generate some synthetic flare samples (xl_max_from_xl_w_data_augmentation_cnn1d_v1)

Paper:
https://ieeexplore.ieee.org/document/9207284
