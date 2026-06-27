import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Standard 7x7 convolution, padding=3 to keep spatial size.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=7, padding=3, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

batch_size = 1
in_channels = 3
out_channels = 32
height = 224
width = 224

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]