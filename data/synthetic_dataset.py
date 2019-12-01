import numpy as np
import torch
import random
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader
random.seed(0)
def create_synthetic_dataset(N, N_input,N_output,sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    X = []
    breakpoints = []
    for k in range(2*N):
        serie = np.array([ sigma*random.random() for i in range(N_input+N_output)])
        i1 = random.randint(1,10)
        i2 = random.randint(10,18)
        j1 = random.random()
        j2 = random.random()
        interval = abs(i2-i1) + random.randint(-3,3)
        serie[i1:i1+1] += j1
        serie[i2:i2+1] += j2
        serie[i2+interval:] += (j2-j1)
        X.append(serie)
        breakpoints.append(i2+interval)
    X = np.stack(X)
    breakpoints = np.array(breakpoints)
    # print breakpoints
    return X[0:N,0:N_input], X[0:N, N_input:N_input+N_output], X[N:2*N,0:N_input], X[N:2*N, N_input:N_input+N_output],breakpoints[0:N], breakpoints[N:2*N]

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, breakpoints):
        super(SyntheticDataset, self).__init__()  
        self.X_input = X_input
        self.X_target = X_target
        self.breakpoints = breakpoints
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx,:,np.newaxis], self.X_target[idx,:,np.newaxis]  , self.breakpoints[idx])

# ECG5000 data; filepath - Change as appropriate
def load_ECG5000(filepath_loc_train, filepath_loc_test):
    train_data = np.loadtxt(filepath_loc_train)
    train_data = np.delete(train_data, 0, axis = 1)
    test_data = np.loadtxt(filepath_loc_test)
    test_data = np.delete(test_data, 0, axis = 1)
    train_bkp = np.asarray([0]*500)
    test_bkp = np.asarray([0]*4500)
    x_train_ip = train_data[0:500, 0:84] 
    x_train_op = train_data[0:500, 84:140]
    x_test_ip = test_data[0:4500, 0:84] 
    x_test_op = test_data[0:4500, 84:140]
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp

# This dataset class is used for every dataset. 
class ECG5000Dataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, breakpoints):
        super(ECG5000Dataset, self).__init__()  
        self.X_input = X_input
        self.X_target = X_target
        self.breakpoints = breakpoints
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        # return (self.X_input[idx,:], self.X_target[idx,:], self.breakpoints[idx])
        return (self.X_input[idx,:,np.newaxis], self.X_target[idx,:,np.newaxis], self.breakpoints[idx])


# Minimum Temperature Dataset
# def load_mintemp(filepath, N):
#     data = pd.read_csv(filepath)
#     data  = list(data['Temp']) 
#     total_data = []
#     for i in range(len(data)-59):
#         a = [data[i:i+60]]
#         total_data.append(a)
#     train_data = total_data[:1000]
#     test_data = total_data[1000:3500]
#     # converting data between -1,1
#     train_data, test_data = np.asarray(train_data), np.asarray(test_data)
#     train_data, test_data = np.squeeze(train_data, axis = 1), np.squeeze(test_data, axis = 1)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     train_data = min_max_scaler.fit_transform(train_data)
#     test_data = min_max_scaler.fit_transform(test_data)
#     # print (train_data.shape, test_data.shape)
 
#     train_bkp = np.asarray([0]*len(train_data))
#     test_bkp = np.asarray([0]*len(test_data))
#     x_train_ip = train_data[:, 0:N] 
#     x_train_op = train_data[:, N:]
#     x_test_ip = test_data[:, 0:N] 
#     x_test_op = test_data[:, N:]        
#     return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp


# Traffic dataset - filepath - Change as appropriate
def load_traffic(filepath):
    data = pd.read_csv(filepath, header = None)
    data = np.asarray(data)
    data = data[:,1]
    total_data = []
    x = 0
    for i in range(104):
        m = [data[x:x+192]]
        x = x + 168
        total_data.append(m)
    total_data = np.asarray(total_data)
    total_data = np.squeeze(total_data)
    k = 24  
    n = len(total_data)
    x_train_ip = total_data[:75, 0:168] 
    x_train_op = total_data[:75, 168:]
    x_test_ip = total_data[75:100, 0:168] 
    x_test_op = total_data[75:100, 168:]
    train_bkp = np.asarray([0]*75)
    test_bkp = np.asarray([0]*25)
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp  


# Wafer Dataset - filepath - Change as appropriate
def load_wafer(filepath_loc_train, filepath_loc_test):
    train_data = np.loadtxt(filepath_loc_train)
    train_data = np.delete(train_data, 0, axis = 1)
    test_data = np.loadtxt(filepath_loc_test)
    test_data = np.delete(test_data, 0, axis = 1)
    train_bkp = np.asarray([0]*1000)
    test_bkp = np.asarray([0]*6000)
    x_train_ip = train_data[0:1000, 0:90] 
    x_train_op = train_data[0:1000, 90:]
    x_test_ip = test_data[:6000, 0:90] 
    x_test_op = test_data[:6000, 90:]
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp
