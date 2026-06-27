import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs standard Dense (regular) 2D Convolution.
    All input channels are connected to all output channels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the Dense Conv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution. Default: 1.
            padding (int): Padding added to input. Default: 0.
        """
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Dense 2D Convolution to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        return self.conv(x)

batch_size = 1
in_channels = 16
out_channels = 64
height = 64
width = 64
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, 1, 1]  # stride=1, padding=1