import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Depthwise Separable Convolution as used in MobileNet.
    Consists of: Depthwise Conv (groups=in_channels) + Pointwise Conv (1x1).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        """
        Initializes Depthwise Separable Convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the depthwise convolution kernel.
            stride (int): Stride of the depthwise convolution. Default: 1.
            padding (int): Padding for depthwise convolution. Default: 0.
        """
        super(Model, self).__init__()
        
        # Depthwise convolution: each channel is convolved independently
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups equals in_channels
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Depthwise Separable Convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

batch_size = 1
in_channels = 16
out_channels = 32
height = 64
width = 64
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, 1, 1]  # stride=1, padding=1