# target is the max in the pred_win

import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import numpy as np
from sklearn.decomposition import PCA
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
from keras.models import Model
import keras.backend as K


''' READ DATA '''
data = np.load('../data/flux_5min_max.npy').astype(np.float32)
data = data+5
data[data<0] = data[data<0]/10

pred_win = 72
lookback = 144

''' making sequence data '''
def sequence_data(data, delay=0, lookback = 288, pred_win=144):
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
        if start_y+pred_win>len(data):
            break
    return np.array(x), np.array(y)

x_seq, y_seq = sequence_data(data, delay=0, lookback = lookback, pred_win=pred_win)

''' splitting the data '''
folds = 5
chain_len = 10000
ind = np.arange(x_seq.shape[0])//chain_len%folds
train = ind!=4
test = ind==4
for i in range(folds):
    print('fold {}, mean {:.3f}, std {:.3f}'.format(i,np.mean(y_seq[ind==i]), np.std(y_seq[ind==i])))


''' Preprocess data with PCA '''
n_comp = 18
pca = PCA(n_components = n_comp)
pca.fit(x_seq[train])

x_seq_train = pca.inverse_transform(pca.transform(x_seq[train]))
#x_seq_train = xy_seq_pca[:,:x_seq.shape[1]]

x_seq_test =  pca.inverse_transform(pca.transform(x_seq[test]))
#x_seq_test = xy_seq_pca[:,:x_seq.shape[1]]

y_seq_train = np.amax(y_seq[train],axis = 1).reshape((-1,1))
y_seq_test = np.amax(y_seq[test],axis = 1).reshape((-1,1))


def display_baselines(x_train, y_train, x_test, y_test):
    bl_pred = np.mean(x_test,axis=1).reshape((-1,1))
    print('bl_pred shape : ', bl_pred.shape, 'y_test shape : ', y_test.shape)
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict mean of lookback MSE =',bl_mse)
    
    bl_pred = np.mean(x_test[:,-pred_win:],axis=1).reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict mean of last pred_win elments in lookback MSE =',bl_mse)
    
    bl_pred = x_test[:,-1].reshape((-1,1))
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict last element in lookback MSE =',bl_mse)

display_baselines(x_seq_train, y_seq_train, x_seq_test, y_seq_test)

n_features = 1
x_seq_train = x_seq_train.reshape((x_seq_train.shape[0], x_seq_train.shape[1], n_features))
x_seq_test = x_seq_test.reshape((x_seq_test.shape[0], x_seq_test.shape[1], n_features))


batch_size_list = [1024]
#batch_size_list = [1024]
lr_list = [0.001]
epochs = 20
dropout_frac = 0.0
l2_penalty=0.000

delay = 0
#pred_win = 72
#lookback = 144
dense_units = pred_win
cn = 1


for batch_size in batch_size_list:
    for lr in lr_list:
        print('batch_size=', batch_size,'lr = ', lr,'cn = ', cn, 'epochs =', epochs,'delay=', delay,'lookback=', lookback,'pred_win=', pred_win, 'components =', n_comp)


        model = models.Sequential()
#model.add(layers.Embedding(max_features, 128, input_length=max_len))
        model.add(layers.Conv1D(32, 7, activation='relu', input_shape = (x_seq_train.shape[1], n_features)))
#model.add(layers.MaxPooling1D(5))
#model.add(layers.Conv1D(256, 7, activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
        model.add(MaxPooling1D(pool_size=7))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(layers.Dense(y_seq_train.shape[1]))
        model.summary()

        model.compile(optimizer=Adam(lr=lr, clipnorm=cn), loss='mse')

# fit model
        history = model.fit(x_seq_train, y_seq_train,  batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_seq_test, y_seq_test))
# demonstrate prediction
        prediction = model.predict(x_seq_test)

        print('prediction shape :', prediction.shape)
        MSE = np.mean(np.square(prediction - y_seq_test))

        print('mse = ', MSE)


        model_path = '/data/sumi/pca/models/cnn1d/'
        pred_path = '/data/sumi/pca/predictions/cnn1d/'
        graph_path = '/data/sumi/pca/graphs/cnn1d/'

        model_name = 'cnn1d'
        str_batch = str(batch_size)
        str_lr = str(lr)
        str_delay = str(pred_win)
        str_look = str(lookback)
        str_comp = str(n_comp)
        str_cn = str(cn)
        model.save(model_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'_'+str_comp+'_'+str_cn+'.hdf5')
        np.save(pred_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'_'+str_comp+'_'+str_cn+'.npy', prediction.astype(np.float32))


        plt.plot(np.arange(prediction.shape[0]), prediction, 'r-', label = 'predicted flux')
        plt.plot(np.arange(prediction.shape[0]), y_seq_test, 'b-', label = 'actual flux')
        plt.legend()
        plt.savefig(graph_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'_'+str_comp+'_'+str_cn+'.png', bbox_inches='tight')

        plt.scatter(y_seq_test,prediction)
        plt.xlabel('actual x-ray flux')
        plt.ylabel('prediction')
        plt.savefig(graph_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'_'+str_comp+'_'+str_cn+'scatter'+'.png', bbox_inches='tight')








