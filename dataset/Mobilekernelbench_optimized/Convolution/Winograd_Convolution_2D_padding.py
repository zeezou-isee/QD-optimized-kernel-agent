import torch
import torch.nn as nn

class Model(nn.Module):
    """
    3x3 convolution (may be accelerated by Winograd in backend), dilation=2.
    """
    def __init__(self, in_ch: int = 16, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=2, dilation=2, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

batch_size = 1
in_channels = 16
out_channels = 16
height = 64
width = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]