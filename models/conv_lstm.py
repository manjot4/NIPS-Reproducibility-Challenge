import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net_Cnn_Lstm(nn.Module):
    def __init__(self, device, in_channels, out_channels, N_input, hidden_dim, batch_size, num_layers, output_size):
        super(Net_Cnn_Lstm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size = 2)
        #LSTM after conv1d
        self.device = device
        self.N_input = N_input
        self.batch_size = batch_size
        self.input_dim_lstm = out_channels*(self.N_input-1)#come from conv1d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim_lstm, self.hidden_dim, num_layers = self.num_layers, batch_first=True) # in- from conv1d, hidd - 128, n_lay - 2
        self.linear = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):
        x = x.reshape((self.batch_size, 1, self.N_input))
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim)).to(self.device)
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim)).to(self.device)
        x = x.unsqueeze(-1)
        x = x.reshape((self.batch_size, 1,  self.input_dim_lstm))
        m, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_dim)
        out = self.linear(h_out)
        out = out.unsqueeze(-1)
        return out
