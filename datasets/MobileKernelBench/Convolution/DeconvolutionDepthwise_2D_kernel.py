import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Depthwise deconvolution (ConvTranspose2d) with kernel_size=5, stride=1, padding=2.
    """
    def __init__(self, in_ch: int , out_ch: int , k: int ,stride, padding, output_padding, dilation):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=in_ch,  # depthwise
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)
# Parameters
batch_size = 1
in_channels = 32
out_channels = 32  # For depthwise, typically in_channels == out_channels == groups
height = 48
width = 48
kernel_size = 5
stride = 1
padding = 2
output_padding = 0
dilation = 1
def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation]