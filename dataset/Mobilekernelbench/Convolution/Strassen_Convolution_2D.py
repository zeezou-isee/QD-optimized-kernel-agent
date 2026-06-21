import torch
import torch.nn as nn

class Model(nn.Module):
    """
    1x1 convolution (conceptually could use Strassen for the GEMM part), stride=2 for downsampling.
    """
    def __init__(self, in_ch: int = 64, out_ch: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

batch_size = 1
in_channels = 64
out_channels = 128
height = 56
width = 56

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]