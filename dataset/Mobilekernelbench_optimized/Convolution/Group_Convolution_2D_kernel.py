import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Grouped convolution with groups equal to half of input channels.
    """
    def __init__(self, in_ch: int = 24, out_ch: int = 24,kernel_size: int =5, groups: int =12, stride: int =1, padding: int =2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

batch_size = 2
def get_inputs():
    x = torch.rand(batch_size, 24, 32, 32)
    return [x]

def get_init_inputs():
    return []


batch_size = 1
in_channels = 24
out_channels = 24
height = 64
width = 64
kernel_size = 5
groups = 12  # Split into 12 groups

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, 1, 2]  # stride=1, padding=2