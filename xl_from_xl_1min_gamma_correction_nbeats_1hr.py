import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
import  matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, LSTM, Convolution2D, Dense, Dropout, Input, Flatten, concatenate, Add, Subtract 
from sklearn import preprocessing
import random
from keras.losses import mean_squared_error
import  matplotlib.pyplot as plt
from exp_filter import *
from box_filter import *
from print_fixed_dec import *
from keras.models import Model
import keras.backend as K

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
    
def inner_block(x,layers,dense_units, backcast_length, forecast_length):
    print('x shape', x.get_shape())
    x1 = Dense(dense_units, kernel_initializer = 'he_normal', activation='relu')(x)
    for i in range(layers-3):
        x1 = Dense(dense_units, kernel_initializer = 'he_normal', activation='relu')(x1)
    backcast = Dense(backcast_length, kernel_initializer = 'glorot_normal')(x1)
    forecast = Dense(forecast_length, kernel_initializer = 'glorot_normal')(x1)
    print('backcast shape :', backcast.get_shape())
    backcast = Subtract()([x, backcast])    
    print('backcast later :', backcast.get_shape())
    return backcast, forecast
    
def stack_block(x,blocks,layers,dense_units,backcast_length, forecast_length):
    b,p = inner_block(x,layers,dense_units,backcast_length, forecast_length)
    for i in range(blocks-1):
        b,p1 = inner_block(b,layers,dense_units, backcast_length, forecast_length)
        p = Add()([p, p1])    
    return b,p

def network(x,stacks,blocks,layers,dense_units, backcast_length, forecast_length):
    b,p = stack_block(x,blocks,layers,dense_units,backcast_length, forecast_length)
    for i in range(stacks-1):
        b,p1 = stack_block(b,blocks,layers,backcast_length, forecast_length)
        p = Add()([p, p1])    
    return b,p        
    
if __name__ == "__main__":    

    batch_size = 128
    lr = 0.001 
    epochs = 5
    dropout_frac = 0.0
    l2_penalty=0.000
    
    delay = 0
    pred_win = 60
    lookback = 360 #pred_win
    dense_units = pred_win
    cn = .5
    
    print('batch_size=', batch_size,'lr = ', lr,'cn = ', cn, 'epochs =', epochs,'delay=', delay,'lookback=', lookback,'pred_win=', pred_win)
    
    #data = np.load('xr_long_1m_data_goes13p_20150121_20190806.npy')
    #data = np.load('xr_long_1m_data_goes15p_20150121_20190806.npy')
    #data = np.load('xr_long_1m_data_goes13_20150121_20190806_gamma33.npy')
    data = np.load('xr_long_1m_data_goes15_20101027_20190806_gamma33.npy')
    print('xr_long_1m_data_goes15_20101027_20190806_gamma33.npy')
    
    x_seq, y_seq = sequence_data(data, delay=delay,lookback=lookback,pred_win=pred_win)
    print(x_seq.shape)
    print(y_seq.shape)
    
    chain_len = 100000
    ind = np.arange(len(x_seq))//chain_len%5
    train = ind<4
    test = ind==4
    
#    features = Input(shape=(lookback,))
#    b,pred = network(features,1,2,4,lookback, pred_win)
#    model = Model(inputs=features, outputs=[b,pred])
#    model.summary()
#        
    
    bl_pred = np.mean(x_seq[test],axis=1).reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_seq[test])**2)
    print('baseline mse 1 =',bl_mse)
    
    bl_pred = np.mean(x_seq[test,-pred_win:],axis=1).reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_seq[test])**2)
    print('baseline mse 2=',bl_mse)
    
    bl_pred = np.mean(x_seq[test,-60:],axis=1).reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_seq[test])**2)
    print('baseline mse 3=',bl_mse)
    
    bl_pred = x_seq[test,-1].reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_seq[test])**2)
    print('baseline mse 4=',bl_mse)
    
    #model.compile(optimizer=SGD(lr=lr, momentum=0.95, nesterov=False), loss='mse')
    
    btr = x_seq[train]*0
    bts = x_seq[test]*0
    p_sum=y_seq[test]*0
    for i in range(1,2):
        features = Input(shape=(lookback,))
        b,pred = network(features,1,2,2,dense_units,lookback, pred_win)
        model = Model(inputs=features, outputs=[b,pred])
        if i==1:
            model.summary()
        
        model.compile(optimizer=Adam(lr=lr, clipnorm=cn), loss='mse')
        #history = model.fit(x_seq[train], [btr,y_seq[train]], batch_size=batch_size, epochs=epochs, 
        #                    verbose=1)
        history = model.fit(x_seq[train], [btr,y_seq[train]], batch_size=batch_size, epochs=epochs, 
                           verbose=1,validation_data=(x_seq[test], [bts,y_seq[test]])) 
        
        [b,p] = model.predict(x_seq[test])
        print('p shape', p.shape)
        p_sum += p
        print('p_sum shape', p_sum.shape)
        mse =  np.mean((p_sum/i-y_seq[test])**2)
        print('error after',i,'iterations:',mse)
    
model_name = 'nbeats'
str_batch = str(batch_size)
str_lr = str(lr)
str_delay = str(pred_win)
str_look = str(lookback)
model.save('/data/sumi/gamma_correction/nbeats_model_pred/nbeats_model/'+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'.hdf5')
np.save('/data/sumi/gamma_correction/nbeats_model_pred/nbeats_pred/'+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'.npy', p.astype(np.float32))
   


