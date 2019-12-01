# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Net_Cnn(nn.Module):

#     def __init__(self,in_channels,out_channels,N_input,output_size):
#         super(Net_Cnn, self).__init__()

#         self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size = 2)
#         # self.mp =  nn.MaxPool1d(2)
#         # self.relu = nn.ReLU()
#         #self.f = nn.Flatten()
#         self.fc1 = nn.Linear(out_channels*(N_input-1),output_size)
#         #self.fc2 = nn.Linear(1,output_size)

#         #
#         # # fully connected layer
#         # self.fc1 = nn.Linear(input_size, hidden_size)
#         # self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):

#         #x = x.view(x.size(0),-1)
#         #x = x.squeeze(-1)

#         x = x.reshape((100, 1, 20))

#         x = self.conv1(x)

#         x = F.relu(x)


#         x = torch.flatten(x, 1)

#         #out = nn.Flatten(out)

#         x = self.fc1(x)
#         #print(x.shape)

#         #out = self.fc2(out)

#         #out.squeeze(-1)

#         #x = F.relu(x)
#         x = x.unsqueeze(-1)

#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_Cnn(nn.Module):

    def __init__(self,in_channels,out_channels,N_input,output_size):
        super(Net_Cnn, self).__init__()

        self.conv1 = nn.Conv1d(in_channels,out_channels,kernel_size = 2)
        # self.mp =  nn.MaxPool1d(2)
        # self.relu = nn.ReLU()
        #self.f = nn.Flatten()
        #self.conv2 = nn.Conv1d(out_channels,10,kernel_size=2)
        self.fc1 = nn.Linear(out_channels*(N_input-1),30)
        self.fc2 = nn.Linear(30,output_size)

        #
        # # fully connected layer
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        #x = x.view(x.size(0),-1)
        #x = x.squeeze(-1)

        x = x.reshape((100, 1, 20))

        x = self.conv1(x)

        x = F.relu(x)
        #print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        #x = F.relu(x)

        x = torch.flatten(x, 1)

        #out = nn.Flatten(out)

        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.shape)

        #out = self.fc2(out)

        #out.squeeze(-1)

        #x = F.relu(x)
        x = x.unsqueeze(-1)

        return x

