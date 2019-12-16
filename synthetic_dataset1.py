import numpy as np
import torch
import random
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import Dataset, DataLoader
random.seed(0)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

# class for ECG5000 data
def load_ECG5000(filepath_loc_train, filepath_loc_test):
    train_data = np.loadtxt(filepath_loc_train)
    train_data = np.delete(train_data, 0, axis = 1)
    test_data = np.loadtxt(filepath_loc_test)
    test_data = np.delete(test_data, 0, axis = 1)
    # Normalizing data
    # scaler = StandardScaler().fit(train_data)
    scaler = MinMaxScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    #doing data preprocessing
#--------
    total_train_data, total_test_data = [], []
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            total_train_data.append(train_data[i][j])
            total_test_data.append(test_data[i][j])
    train_max, train_min, test_max, test_min = max(total_train_data), min(total_train_data), max(total_test_data), min(total_test_data)
    print train_max, train_min, test_max, test_min 
    print len(total_train_data)
    # print (train_data.shape, test_data.shape)
    # # print train_data[:5]
    # # print test_data[:5]
    # plt.figure()
    # plt.plot(train_data[0])
    # plt.show()
    # plt.figure()
    # plt.plot(test_data[0])
    # plt.show()
# --------

    train_bkp = np.asarray([0]*500)
    test_bkp = np.asarray([0]*4500)
    x_train_ip = train_data[0:500, 0:84] 
    x_train_op = train_data[0:500, 84:140]
    x_test_ip = test_data[0:4500, 0:84] 
    x_test_op = test_data[0:4500, 84:140]
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp

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

def load_mintemp(filepath, N):
    data = pd.read_csv(filepath)
    data  = list(data['Temp']) 
    total_data = []
    for i in range(len(data)-59):
        a = [data[i:i+60]]
        total_data.append(a)
    train_data = total_data[:1000]
    test_data = total_data[1000:3500]
    # converting data between -1,1
    train_data, test_data = np.asarray(train_data), np.asarray(test_data)
    train_data, test_data = np.squeeze(train_data, axis = 1), np.squeeze(test_data, axis = 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    test_data = min_max_scaler.fit_transform(test_data)
    # print (train_data.shape, test_data.shape)
 
    train_bkp = np.asarray([0]*len(train_data))
    test_bkp = np.asarray([0]*len(test_data))
    x_train_ip = train_data[:, 0:N] 
    x_train_op = train_data[:, N:]
    x_test_ip = test_data[:, 0:N] 
    x_test_op = test_data[:, N:]        
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp



def load_traffic(filepath):
    data = pd.read_csv(filepath, header = None)
    data = np.asarray(data)
    data = np.mean(data, axis=1)
    data = list(data)
    # windowing
    total_data = []
    for i in range(len(data)-191):
        x = [data[i:i+192]]
        total_data.append(x)    
    total_data = np.asarray(total_data)
    total_data = np.squeeze(total_data)
    k = 24  
    n = len(total_data)
    # tr = int(0.7)
    x_train_ip = total_data[:500, 0:168] 
    x_train_op = total_data[:500, 168:]
    # print x_train_ip.shape, x_train_op.shape

    x_test_ip = total_data[11000:15000, 0:168] 
    x_test_op = total_data[11000:15000, 168:]
    # print x_test_ip.shape, x_test_op.shape
    train_bkp = np.asarray([0]*10000)
    test_bkp = np.asarray([0]*4000)
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp    

# Wafer Dataset - filepath - Change as appropriate
def load_wafer(filepath_loc_train, filepath_loc_test):
    train_data = np.loadtxt(filepath_loc_train)
    train_data = np.delete(train_data, 0, axis = 1)
    test_data = np.loadtxt(filepath_loc_test)
    test_data = np.delete(test_data, 0, axis = 1)
    #doing data preprocessing
#--------
    total_train_data, total_test_data = [], []
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            total_train_data.append(train_data[i][j])
            total_test_data.append(test_data[i][j])
    train_max, train_min, test_max, test_min = max(total_train_data), min(total_train_data), max(total_test_data), min(total_test_data)
    print train_max, train_min, test_max, test_min 
    print len(total_train_data)
    # plt.figure()
    # plt.plot(train_data)
    # plt.show()
    # plt.figure()
    # plt.plot(test_data)
    # plt.show()

# --------   
    train_bkp = np.asarray([0]*1000)
    test_bkp = np.asarray([0]*6000)
    x_train_ip = train_data[0:1000, 0:90] 
    x_train_op = train_data[0:1000, 90:]
    x_test_ip = test_data[:6000, 0:90] 
    x_test_op = test_data[:6000, 90:]
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp

