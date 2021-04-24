import numpy as np

def exp_filter(data, n, forward= True):
    alpha = 1/n
    beta = 1-alpha
    data_f = np.zeros_like(data)
    if forward:
        data_f[0]= data[0]
        for i in range(1,len(data)):
            data_f[i] = min([data_f[i-1]*2, data[i]])*alpha+data_f[i-1]*beta
    else:
        n = len(data)
        data_f[n-1]= data[n-1]
        for i in range(n-2,-1,-1):
            data_f[i] = data[i]*alpha+data_f[i+1]*beta
    return data_f

