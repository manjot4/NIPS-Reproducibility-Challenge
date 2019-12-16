import numpy as np
# from  data import synthetic_dataset 
from synthetic_dataset import create_synthetic_dataset, SyntheticDataset, load_ECG5000, ECG5000Dataset, load_mintemp, load_traffic, load_wafer
import torch
from seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
from scipy.stats import wasserstein_distance
from fnn import *
from conv_lstm import *
# from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
# from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
# from loss.dilate_loss import dilate_loss
import time
# start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

# parameters
batch_size = 100
N = 500
N_input = 20
N_output = 20  
sigma = 0.01
gamma = 0.01

# ECG5000
loc = '/Users/manjotsingh/desktop/NIPS_challenge/ECG5000/' #change as appropriate
filepath_train = loc + 'ECG5000_TRAIN.txt'
filepath_test = loc + 'ECG5000_TEST.txt'

# # Load synthetic dataset
# X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)
# dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
# dataset_test  = SyntheticDataset(X_test_input,X_test_target, test_bkp)
# trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
# testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)

N_input = 84
N_output = 56
# Load ECG dataset
x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp = load_ECG5000(filepath_train, filepath_test)
# print (x_train_ip.shape, x_train_op.shape, x_test_ip.shape, x_test_op.shape)
dataset_train = ECG5000Dataset(x_train_ip, x_train_op, train_bkp)
dataset_test  = ECG5000Dataset(x_test_ip, x_test_op, test_bkp)
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)

# Min Temp
# N_input = 30
# N_output = 30
# loc = '/Users/manjotsingh/desktop/NIPS_challenge/DILATE/min_temp.txt'
# # Load Min Temp dataset
# x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp = load_mintemp(loc, N_output)
# # print (x_train_ip.shape, x_train_op.shape, x_test_ip.shape, x_test_op.shape)
# dataset_train = ECG5000Dataset(x_train_ip, x_train_op, train_bkp)
# dataset_test  = ECG5000Dataset(x_test_ip, x_test_op, test_bkp)
# trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
# testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)



# # Traffic Analysis

# N_input = 168
# N_output = 24
# loc = '/Users/manjotsingh/desktop/NIPS_challenge/traffic.txt'
# # Load Min Temp dataset
# x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp = load_traffic(loc)
# # print (x_train_ip.shape, x_train_op.shape, x_test_ip.shape, x_test_op.shape)
# dataset_train = ECG5000Dataset(x_train_ip, x_train_op, train_bkp)
# dataset_test  = ECG5000Dataset(x_test_ip, x_test_op, test_bkp)
# trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
# testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)



# # Wafer
# loc = '/Users/manjotsingh/desktop/NIPS_challenge/Wafer/' #change as appropriate
# filepath_train = loc + 'Wafer_TRAIN.txt'
# filepath_test = loc + 'Wafer_TEST.txt'
# N_input = 90
# N_output = 62
# # Load ECG dataset
# x_train_ip, x_train_op, x_test_ip, x_test_op, train_bkp, test_bkp = load_wafer(filepath_train, filepath_test)
# # print (x_train_ip.shape, x_train_op.shape, x_test_ip.shape, x_test_op.shape)
# dataset_train = ECG5000Dataset(x_train_ip, x_train_op, train_bkp)
# dataset_test  = ECG5000Dataset(x_test_ip, x_test_op, test_bkp)
# trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
# testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)


 
# training losses over all the epochs
tr_loss = []
tr_loss_shape = []
tr_loss_temp = []

def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    huber_loss = torch.nn.SmoothL1Loss()
    
    for epoch in range(epochs): 
        # start_time = time.time()
        l, l_shape, l_temp, n_tr_steps = 0, 0, 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]                     

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type == 'huber'):
                loss = huber_loss(target, outputs)

            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'): 
                # print (outputs.size(), target.size())   
                # loss = dilate_loss(target,outputs,alpha, gamma, device)   
                # print (loss.grad_fn)
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
            

            optimizer.zero_grad()
            # with torch.autograd.profiler.profile(use_cuda=False) as prof:
            #     loss.backward()
            # print(prof)
            # print (list(net.parameters())[0])
            loss.backward()
            optimizer.step()

        # adding training losses
            l += loss.item() 
            l_shape += loss_shape.item()
            l_temp += loss_temporal.item()
            n_tr_steps += 1
        tr_loss.append(l/float(n_tr_steps))
        tr_loss_shape.append(l_shape/float(n_tr_steps))
        tr_loss_temp.append(l_temp/float(n_tr_steps))    
        
        # end_time = time.time()
        # print ("time_taken: ", end_time - start_time, " seconds")
        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                eval_model(net,testloader, gamma,verbose=1)

  
def eval_model(net,loader, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

            loss_dtw += dtw(target_k_cpu,output_k_cpu)
            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                       
            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)            
                        
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    print( ' Eval mse= ', np.array(losses_mse).mean() ,' dtw= ',np.array(losses_dtw).mean() ,' tdi= ', np.array(losses_tdi).mean()) 



# HUBER LOSS
encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
train_model(net_gru_mse,loss_type='huber',learning_rate=0.001, epochs=700, gamma=gamma, print_every=50, eval_every=50,verbose=1)

# # DILATE LOSS
# encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
# decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
# net_gru_dilate = Net_GRU(encoder,decoder, N_output, device).to(device)
# train_model(net_gru_dilate,loss_type='dilate',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)

# # MSE LOSS
# encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
# decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
# net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
# train_model(net_gru_mse,loss_type='mse',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)


# Conv-net
# net_cnn = Net_Cnn(device, in_channels=1, out_channels=30, N_input=N_input, hidden_dim=128, batch_size=batch_size, num_layers=1, output_size=N_output).to(device)
# train_model(net_cnn,loss_type='dilate',learning_rate=0.001, epochs=700, gamma=gamma, print_every=50, eval_every=50,verbose=1)


# plot training losses - 
# plt.figure()
# plt.plot(tr_loss)
# plt.show()
# plt.figure()
# plt.plot(tr_loss_shape)
# plt.show()
# plt.figure()
# plt.plot(tr_loss_temp)
# plt.show()



# # Linear Layer------- Run this 
# net_fnn_dilate = Net_Fnn(N_input, 128, N_output, device).to(device)
# train_model(net_fnn_dilate,loss_type='dilate',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)

# net_fnn_mse = Net_Fnn(N_input,128, N_output, device).to(device)
# train_model(net_fnn_mse,loss_type='mse',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)


# # Visualize results
# gen_test = iter(testloader)
# test_inputs, test_targets, breaks = next(gen_test)

# test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
# test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
# criterion = torch.nn.MSELoss()

# nets = [net_gru_mse,net_gru_dilate]

# # for ind in range(1,51):
# #     plt.figure()
# #     plt.rcParams['figure.figsize'] = (17.0,5.0)  
# #     k = 1
# #     for net in nets:
# #         pred = net(test_inputs).to(device)

# #         input = test_inputs.detach().cpu().numpy()[ind,:,:]
# #         target = test_targets.detach().cpu().numpy()[ind,:,:]
# #         preds = pred.detach().cpu().numpy()[ind,:,:]

# #         plt.subplot(1,3,k)
# #         plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
# #         plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
# #         plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ])  ,label='prediction',linewidth=3)       
# #         plt.xticks(range(0,40,2))
# #         plt.legend()
# #         k = k+1

# #     plt.show()


# end_time = time.time()
# print ("time_taken: ", end_time - start_time, " seconds")



# need to run again - ECG, Wafer - 10 times because of data distribution
# see the distribution of my errors
# then decide if to use the loss or not
# run the code many times
# linvpy library
# ran Huber loss on synthetic and ECG5000