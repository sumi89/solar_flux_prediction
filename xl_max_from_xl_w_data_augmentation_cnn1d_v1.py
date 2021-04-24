# target is the max in the pred_win
# data augmentation is implemented here

import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
import collections

''' READ DATA '''
#data = np.load('/Users/sumi/python/research/pca_max/flux_5min_max.npy').astype(np.float32)
data = np.load('../data/flux_5min_max.npy').astype(np.float32)
data = data+5
data[data<0] = data[data<0]/10
print('min of log data : ', min(data))
print('max of log data : ', max(data))

pred_win = 12
lookback = pred_win*2

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
        y.append(np.max(data[start_y:start_y+pred_win]))
        start_x, start_y =  start_x+1, start_y+1
        if start_y+pred_win>len(data):
            break
    return np.array(x), np.array(y)

x_seq, y_seq = sequence_data(data, delay=0, lookback = lookback, pred_win=pred_win)

''' splitting the data '''
folds = 5
chain_len = 10000
ind = np.arange(x_seq.shape[0])//chain_len%folds
train = ind!=2
test = ind==2
for i in range(folds):
    print('fold {}, mean {:.3f}, std {:.3f}'.format(i,np.mean(y_seq[ind==i]), np.std(y_seq[ind==i])))


x_seq_train = x_seq[train]
x_seq_test = x_seq[test]
y_seq_train = y_seq[train]
y_seq_test = y_seq[test]

print('x_seq_train shape : ', x_seq_train.shape)
print('x_seq_test shape : ', x_seq_test.shape)
print('y_seq_train shape : ', y_seq_train.shape) 
print('y_seq_test shape : ', y_seq_test.shape)

print('min of scaled data : ', min(y_seq_train))
print('max of scaled data : ', max(y_seq_train))

def display_baselines(x_train, y_train, x_test, y_test):
    bl_pred = np.mean(x_test,axis=1)
    print('bl_pred shape : ', bl_pred.shape, 'y_test shape : ', y_test.shape)
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict mean of lookback MSE =',bl_mse)
    
    bl_pred = np.mean(x_test[:,-pred_win:],axis=1)
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict mean of last pred_win elments in lookback MSE =',bl_mse)
    
    bl_pred = x_test[:,-1]
    bl_mse =  np.mean((bl_pred-y_test)**2)
    print('Predict last element in lookback MSE =',bl_mse)

display_baselines(x_seq_train, y_seq_train, x_seq_test, y_seq_test)


############ DATA AUGMENTATION PART ################

# covert data to original space
x_seq_train_cp = x_seq[train]
y_seq_train_cp = y_seq[train]

x_seq_train_cp[x_seq_train_cp<0] = (x_seq_train_cp[x_seq_train_cp<0]*10)
x_seq_train_cp = x_seq_train_cp-5
x_seq_train_cp = 10**x_seq_train_cp
y_seq_train_cp[y_seq_train_cp<0] = (y_seq_train_cp[y_seq_train_cp<0]*10)
y_seq_train_cp = y_seq_train_cp-5
y_seq_train_cp = 10**y_seq_train_cp

print('min of data in original space : ', min(y_seq_train_cp))
print('max of data in original space : ', max(y_seq_train_cp))

# check the number of higher fluxes
hf = np.sum(y_seq_train>=0) # number of higher fluxes (flux>=0)
lf = np.sum(y_seq_train<0) # number of lower fluxes (flux<0)
print('number of samples with higher fluxes (flux>=0) = ', hf)
print('number of samples with lower fluxes (flux<0) = ', lf)

req_samples = hf*(lf//hf//10*10)
print('number of samples generating = ', req_samples)


# indices of higher fluxes (flux >= 0)
ind_hf = np.where(y_seq_train>=0)

# x_seq with higher fluxes (flux >= 0)
x_seq_hf = x_seq_train_cp[ind_hf]
# y_seq with higher fluxes (flux >= 0)
y_seq_hf = y_seq_train_cp[ind_hf]


x_seq_gen = []
y_seq_gen = []

for i in range(req_samples):
    w = random.random()
    random_list = []
    for i in range(2):
        random_list.append(random.randint(0,hf-1))
    #print('random_list : ', random_list, 'w = ', w)
    x_seq_n = x_seq_hf[random_list[0]]*w + x_seq_hf[random_list[1]]*(1-w)
    y_seq_n = y_seq_hf[random_list[0]]*w + y_seq_hf[random_list[1]]*(1-w)
    x_seq_gen.append(x_seq_n)
    y_seq_gen.append(y_seq_n)


x_seq_gen_arr = np.array(x_seq_gen)
y_seq_gen_arr = np.array(y_seq_gen)

x_seq_aug = np.vstack((x_seq_train_cp, x_seq_gen_arr))
y_seq_aug = np.append(y_seq_train_cp, y_seq_gen_arr)

############## DATA AUGMENTATION PART END ##############

# scaling the augmented data
x_seq_train_aug_scale = np.log10(x_seq_aug)+5
x_seq_train_aug_scale[x_seq_train_aug_scale<0] = x_seq_train_aug_scale[x_seq_train_aug_scale<0]/10

y_seq_train_aug_scale = np.log10(y_seq_aug)+5
y_seq_train_aug_scale[y_seq_train_aug_scale<0] = y_seq_train_aug_scale[y_seq_train_aug_scale<0]/10

print('min of data in log space after augmentation and scaling : ', min(y_seq_train_aug_scale))
print('max of data in log space after augmentation and scaling : ', max(y_seq_train_aug_scale))

n_features = 1
x_seq_train_aug_scale = x_seq_train_aug_scale.reshape((x_seq_train_aug_scale.shape[0], x_seq_train_aug_scale.shape[1], n_features))
x_seq_test = x_seq_test.reshape((x_seq_test.shape[0], x_seq_test.shape[1], n_features))


batch_size = 1024
lr = 0.001
epochs = 20
#dropout_frac = 0.0
#l2_penalty=0.000

delay = 0
#pred_win = 72
#lookback = 144
#dense_units = pred_win
cn = 1

print('batch_size=', batch_size,'lr = ', lr,'cn = ', cn, 'epochs =', epochs,'delay=', delay,'lookback=', lookback,'pred_win=', pred_win)


model = models.Sequential()
#model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 3, activation='relu', input_shape = (x_seq_train_aug_scale.shape[1], n_features)))
#model.add(layers.MaxPooling1D(5))
#model.add(layers.Conv1D(256, 7, activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
model.add(MaxPooling1D(pool_size=7))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=Adam(lr=lr, clipnorm=cn), loss='mse')

# fit model
history = model.fit(x_seq_train_aug_scale, y_seq_train_aug_scale,  batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_seq_test, y_seq_test))
# demonstrate prediction
pred = model.predict(x_seq_test)

print('prediction shape :', pred.shape)
MSE = np.mean(np.square(pred - y_seq_test.reshape(-1,1)))

print('mse = ', MSE)

pred_path = '/data/sumi/pca/predictions/cnn1d/'
graph_path = '/data/sumi/pca/graphs/cnn1d/'

model_name = 'cnn1d_aug_x'
str_batch = str(batch_size)
str_lr = str(lr)
str_delay = str(pred_win)
str_look = str(lookback)
#str_comp = str(n_comp)
#model.save(model_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'_'+str_comp+'.hdf5')
np.save(pred_path+model_name+'_'+str_batch+'_'+str_lr+'_'+str_look+'_'+str_delay+'.npy', pred.astype(np.float32))


from collections import Counter


pred_int = pred.copy()
for i in range(pred.shape[0]):
    if pred[i] >= 0:
        pred_int[i] = np.round(pred[i]).astype(int)
    else:
        pred_int[i] = -1

y_seq_test_int = y_seq_test
for i in range(y_seq_test.shape[0]):
    if y_seq_test[i] >= 0:
        y_seq_test_int[i] = np.round(y_seq_test[i]).astype(int)
    else:
        y_seq_test_int[i] = -1

y_seq_test_int = y_seq_test_int.astype(int)
pred_int = pred_int.astype(int)

print('Counter:',collections.Counter(y_seq_test_int.reshape(-1)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_seq_test_int, pred_int)
print('Confusion matrix :\n', cm)


tp0 = np.sum(cm[1:4,1:4])
fn0 = np.sum(cm[1:4,0])
fp0 = cm[0,1]
tn0 = cm[0,0]

precision0 = tp0/(tp0+fp0)
recall0 = tp0/(tp0+fn0)
print('Precision for M and X class : ', precision0, '\nRecall for M and X class : ', recall0)


tp1 = np.sum(cm[2:4, 2:4])
fn1 = np.sum(cm[2:4,0:2])
fp1 = np.sum(cm[0:2,2:4])
tn1 = np.sum(cm[0:2,0:2])

precision1 = tp1/(tp1+fp1)
recall1 = tp1/(tp1+fn1)
print('Precision for X class : ', precision1, '\nRecall for X class : ', recall1)



from sklearn.metrics import roc_curve
from sklearn.metrics import auc

actual_mx = (y_seq_test_int >=0).astype(int)
fpr_mx, tpr_mx, threshold_mx = roc_curve(actual_mx, pred_int)
roc_auc_mx = auc(fpr_mx, tpr_mx)
print("AUC_g_mx", roc_auc_mx)
plt.plot(fpr_mx, tpr_mx)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for M and X-class")
plt.savefig(graph_path+'ROC_M_X.png')



actual_x = (y_seq_test_int >=1).astype(int)
fpr_x, tpr_x, threshold_x = roc_curve(actual_x, pred_int)
roc_auc_x = auc(fpr_x, tpr_x)
print("AUC_g_x", roc_auc_x)
plt.plot(fpr_x, tpr_x)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("ROC curve for X-class")
plt.savefig(graph_path+'ROC_X.png')




























































