import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Group Convolution.
    Input channels are divided into groups, each group is convolved independently.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int, 
                 stride: int = 1, padding: int = 0):
        """
        Initializes the Group Conv2d layer.

        Args:
            in_channels (int): Number of input channels (must be divisible by groups).
            out_channels (int): Number of output channels (must be divisible by groups).
            kernel_size (int): Size of the convolution kernel.
            groups (int): Number of groups for grouped convolution.
            stride (int): Stride of the convolution. Default: 1.
            padding (int): Padding added to input. Default: 0.
        """
        super(Model, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Convolution to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        return self.conv(x)

batch_size = 1
in_channels = 16
out_channels = 32
height = 64
width = 64
kernel_size = 3
groups = 4  # Split into 4 groups

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, 1, 1]  # stride=1, padding=1