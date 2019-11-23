import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_Fnn(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,device):
        super(Net_Fnn, self).__init__()

        # fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # def forward(self, x):
    #
    #     x = x.view(x.size(0),-1)
    #     print(x.size())
    #     x = self.fc1(x)
    #     # x = self.relu(x)
    #     print('cp6')
    #     print(x.size())
    #     #x = F.relu(self.fc2(x))
    #     x = F.relu(x)
    #     x = self.fc2(x)
    #     x = F.relu(x)
    #     x = x.unsqueeze(-1)
    #     print(x.size())
    #
    #     return x

    def forward(self, x):

        x = x.view(x.size(0),-1)
        #x = x.squeeze(-1)

        x = self.fc1(x)
        x = F.relu(x) #do we need relu?
        x = self.fc2(x)
        #x = F.relu(x)
        x = x.unsqueeze(-1)

        return x

