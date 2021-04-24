from exp_filter import *
import  matplotlib.pyplot as plt

def normalize(x):
    mx = np.amax(x)
    mn = np.amin(x)
    return (x-mn)/(mx-mn)
        
def remove_invalid(x,min_val=1e-9):
    for i in range(1,len(x)):
        if x[i]<=min_val:
            x[i]=x[i-1]
    return x

def preprocess(signal, fw=2016, gamma=0.2,min_val=1e-9):
    gmx = np.copy(signal)
    
    # if reading is less than minimum valid value, replace by previous reading 
    remove_invalid(gmx,min_val=1e-9)

    # subtract long-term trend, getting rid of negative values. 
    # exp_filter crops the new values added to the trend to 2*trend
    trend = exp_filter(gmx,fw)
    gmx = np.maximum(0,gmx-trend*.9)
    #gmx = gmx-trend
    #  Normalize to the 0-1 range
    gmx = normalize(gmx)
        
    # Apply gamma scaling
    gmx = gmx**gamma
    #return gmx, trend, mx, mn
    return gmx
    
def plot_by_parts(x,p=10):
    for i in range(p):
        plt.figure()
        start = int(i/p*len(x))
        end = int((i+1)/p*len(x))
        plt.plot(x[start:end],'b-')
   

if __name__ == "__main__":   
    plt.close('all') 
    g13 = np.load('xr_long_1m_data_goes13_20150121_20190806.npy').reshape(-1)
    g15 = np.load('xr_long_1m_data_goes15_20101027_20190806.npy').reshape(-1)
    
    g13p = preprocess(g13)
    print(np.mean(g13))
    print(np.mean(g13p))
    
    g15p = preprocess(g15)
    print(np.mean(g15p))
    np.save('xr_long_1m_data_goes13_20150121_20190806_gamma20.npy',g13p)
    np.save('xr_long_1m_data_goes15_20101027_20190806_gamma20.npy',g15p)
    
    #plot_by_parts(g13p)
    
    
    #plot_by_parts(normalize(remove_invalid(g13))**.33)
    
'''
# if reading is less than 1e-9, replace by previous reading 
for i in range(1,len(g15)):
    if g15[i]<=1e-9:
        g15[i]=g15[i-1]
    if g13[i]<=1e-9:
        g13[i]=g13[i-1]  

# if satellites cover simultaneous time windows, take larger reading of the two satellites
gmx = np.maximum(g13,g15)

# subtract long-term trend, getting rid of negative values. 
# exp_filter crops the new values added to the trend to 2*trend
gmx = np.maximum(0,gmx-exp_filter(gmx,12*24*7))

#  Normalize to the 0-1 range
mx = np.amax(gmx)
mn = np.amin(gmx)
gmx = (gmx-mn)/(mx-mn)

#  Apply gamma scaling
gmx = gmx**gamma

plt.close('all')  
slices = 10
for i in range(slices):
    plt.figure(i)
    start = int(i/slices*len(gmx))
    end = int((i+1)/slices*len(gmx))
    plt.plot(gmx[start:end],'b-')
'''
