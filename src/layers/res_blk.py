import torch.nn as nn
import torch.nn.functional as F
from src.layers.conv import conv1x1, dwconv3x3, pconv3x3
    
class DWConvRB(nn.Module):
    def __init__(self, N=192, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            dwconv3x3(N),
            conv1x1(N, middle_ch),
            act(),
            conv1x1(middle_ch, N),
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out
    
class PConvRB(nn.Module):
    def __init__(self, N=192, partial_ratio=4, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        N1 = N // partial_ratio
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            pconv3x3(N, N1),
            conv1x1(N, middle_ch),
            act(inplace=True),
            conv1x1(middle_ch, N),
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out
