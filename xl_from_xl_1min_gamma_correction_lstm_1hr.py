import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
import  matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, LSTM, Convolution2D, Dense, Dropout, Input, Flatten, concatenate, Add, Subtract 
#from sklearn import preprocessing
import random
from keras.losses import mean_squared_error
import  matplotlib.pyplot as plt
from exp_filter import *
from box_filter import *
from print_fixed_dec import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import EarlyStopping

def sequence_data(seq, delay=0, lookback = 288, pred_win=144):  # x and y are in different length... for a sequence x, predicting sequence y..
    x = []
    y = []
    start_x = 0
    end_x =  lookback
    start_y = lookback+delay
    end_y =  start_y+pred_win
    while True:
        x.append(data[start_x:start_x+lookback])
        y.append(data[start_y:start_y+pred_win])
        start_x, start_y =  start_x+1, start_y+1
        if start_y+pred_win>len(seq):
            break
    return np.array(x), np.array(y)
    
batch_size = 128    
lr = 0.001 
epochs = 5
dropout_frac = 0.0
l2_penalty=0.000
    
delay = 0
pred_win = 60
lookback = pred_win
dense_units = pred_win
cn = .5

print('batch_size=', batch_size,'lr = ', lr,'cn = ', cn, 'epochs =', epochs,'delay=', delay,'lookback=', lookback,'pred_win=', pred_win)


data = np.load('xr_long_1m_data_goes15_20101027_20190806_gamma33.npy')   

x_seq, y_seq = sequence_data(data, delay=delay,lookback=lookback,pred_win=pred_win)
print('x_seq shape : ',x_seq.shape)
print('y_seq shape : ', y_seq.shape)
    
chain_len = 100000
ind = np.arange(len(x_seq))//chain_len%5
train = ind<4
test = ind==4
print('x_seq_tr_shape', x_seq[train].shape)
print('x_seq ts shape : ', x_seq[test].shape)

n_features = 1
x_seq = x_seq.reshape((x_seq.shape[0], n_features, x_seq.shape[1]))

model = models.Sequential()
model.add(LSTM(200, input_shape=(x_seq.shape[1], x_seq.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_seq.shape[1]))
model.summary()

model.compile(optimizer=Adam(lr=lr, clipnorm=cn), loss='mse')
# fit model
history = model.fit(x_seq[train], y_seq[train], epochs=epochs, batch_size=batch_size, validation_data=(x_seq[test], y_seq[test]),verbose=1, shuffle=False) 
                   # callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=1, shuffle=False)
# demonstrate prediction
prediction = model.predict(x_seq[test])

print('prediction shape :', prediction.shape)
MSE = np.mean(np.square(prediction - y_seq[test]))

print('mse = ', MSE)

model_name = 'lstm'
str_batch = str(batch_size)
str_lr = str(lr)
str_delay = str(pred_win)
str_look = str(lookback)
model.save('/data/sumi/gamma_correction/lstm_model_pred/lstm_model/'+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'.hdf5')
np.save('/data/sumi/gamma_correction/lstm_model_pred/lstm_pred/'+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'.npy', prediction.astype(np.float32))
