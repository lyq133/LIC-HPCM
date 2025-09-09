import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch):
    """3x3 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

def conv1x1(in_ch, out_ch):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
    
def conv2x2_down(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

def deconv2x2_up(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, output_padding=0, padding=0)

def conv4x4_down(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

def deconv4x4_up(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, output_padding=0, padding=1)

def dwconv3x3(ch):
    return nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch)

class pconv3x3(nn.Module):
    def __init__(self, N, N1):
        super().__init__()
        self.N = N
        self.N1 = N1
        self.pconv = nn.Conv2d(self.N1, self.N1, 3, 1, 1)
    
    def forward(self, x):
        x1, x2 = torch.split(x, [self.N1, self.N-self.N1], dim=1)
        x1 = self.pconv(x1)
        x = torch.cat((x1, x2), 1)
        return x   
    
    