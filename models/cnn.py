# Convolutional model followed by a inear layer. 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_Cnn(nn.Module):

    def __init__(self,in_channels,out_channels,N_input,output_size):
        super(Net_Cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size = 2)
        self.fc1 = nn.Linear(out_channels*(N_input-1),30)
        self.fc2 = nn.Linear(30,output_size)

    def forward(self, x):
        x = x.reshape((100, 1, 20))
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.unsqueeze(-1)
        return x

