import torch
from torch import nn
import torch.nn.functional as F

class CircularConv1dSame(nn.Module):
    def __init__(self, in_, out_, kss):
        super().__init__()
        self.kss = kss
        self.conv1d = nn.Conv1d(in_, out_, kss, padding=0, padding_mode='zeros', bias=False)
        self.bn1d = nn.BatchNorm1d(out_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()

    def circular_pad1d(self, x):
        pad1d_size = (self.kss // 2, self.kss // 2)
        #x = x.unsqueeze(1)
        x = F.pad(x, pad1d_size, 'circular')
        #x = x.squeeze(1)
        return x

    def forward(self, x):
        x = self.circular_pad1d(x)
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.relu(x)
        return x

class GlobalPool1d(nn.Sequential):
    def __init__(self, average_type='average'):
        layers = []
        if average_type == 'average':
            layers.append(nn.AdaptiveAvgPool1d(1))
        elif average_type == 'max':
            layers.append(nn.AdaptiveMaxPool1d(1))
        layers.append(nn.Flatten())
        super().__init__(*layers)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.f = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.h = nn.Conv1d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B x C x L)
        returns :
            out : self attention feature maps
        """
        B, C, L = x.size()

        f = self.f(x) # B x C x L
        g = self.g(x) # B x C x L
        h = self.h(x) # B x C x L

        att = torch.bmm(f.permute(0, 2, 1), g) # B x L x L
        att = self.softmax(att)

        self_att = torch.bmm(h, att) # B x C x L

        out = self_att + x
        return out