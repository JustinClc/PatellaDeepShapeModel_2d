import torch
from torch import nn
import torch.nn.functional as F

class CircularConv1dSame(nn.Sequential):
    def __init__(self, in_, out_, kss):
        layers = []
        layers.append(nn.Conv1d(in_, out_, kss, padding=kss// 2, padding_mode='zeros', bias=False))
        layers.append(nn.BatchNorm1d(out_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())
        super().__init__(*layers)

class GlobalPool1d(nn.Sequential):
    def __init__(self, average_type='average'):
        layers = []
        if average_type == 'average':
            layers.append(nn.AdaptiveAvgPool1d(1))
        elif average_type == 'max':
            layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(nn.Flatten())
        super().__init__(*layers)

class FCN(nn.Module):
    def __init__(self, hidden_layer=[128,256,128], kernel_size=[7,5,3]):
        super().__init__()
        self.convblock1 = CircularConv1dSame(2, hidden_layer[0], kernel_size[0])
        self.convblock2 = CircularConv1dSame(hidden_layer[0], hidden_layer[1], kernel_size[1])
        self.convblock3 = CircularConv1dSame(hidden_layer[1], hidden_layer[2], kernel_size[2])
        self.gap = GlobalPool1d(average_type='average')
        self.fc = nn.Linear(hidden_layer[2], 1, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2).to(dtype=torch.float)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        return x

