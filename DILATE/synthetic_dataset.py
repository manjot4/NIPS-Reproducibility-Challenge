import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
# random.seed(0)
# https://archive.ics.uci.edu/ml/datasets/PEMS-SF
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
# https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
# https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
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
    train_bkp = np.asarray([0]*2500)
    test_bkp = np.asarray([0]*2500)
    x_train_ip = train_data[0:2500, 0:84] 
    x_train_op = train_data[0:2500, 84:140]
    x_test_ip = test_data[2500:5000, 0:84] 
    x_test_op = test_data[2500:5000, 84:140]
    return x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp

# need to see what exactly to return, might plot a graph to understand data more 
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