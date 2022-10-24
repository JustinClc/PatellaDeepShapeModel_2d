import torch
from torch import nn
import torch.nn.functional as F
from layers import CircularConv1dSame, GlobalPool1d


class CircularFCN(nn.Module):
    def __init__(self, hidden_layer=[128,256,128], kernel_size=[7,5,3]):
        super().__init__()
        self.convblock1 = CircularConv1dSame(2, hidden_layer[0], kernel_size[0])
        self.convblock2 = CircularConv1dSame(hidden_layer[0], hidden_layer[1], kernel_size[1])
        self.convblock3 = CircularConv1dSame(hidden_layer[1], hidden_layer[2], kernel_size[2])
        self.classifier = nn.Sequential(GlobalPool1d(average_type='average'),
                                        nn.Linear(hidden_layer[2], 1, bias=True),
                                        nn.Sigmoid())
                                        #nn.Flatten())

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2).to(dtype=torch.float)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.classifier(x)
        x = torch.flatten(x)
        return x