import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model using 3x3 convolutions which are optimized by Winograd algorithm in cuDNN.
    Winograd reduces multiplications for small kernels (especially 3x3).
    
    For F(2x2, 3x3): reduces from 36 to 16 multiplications.
    cuDNN automatically selects Winograd when beneficial.
    """
    def __init__(self, in_channels: int, out_channels: int, use_winograd_hint: bool = True):
        """
        Initializes Conv2d layer optimized for Winograd algorithm.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_winograd_hint (bool): If True, use 3x3 kernel (Winograd-friendly).
        """
        super(Model, self).__init__()
        
        # 3x3 convolution is optimal for Winograd
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,  # Winograd is most efficient for 3x3
            stride=1,       # Stride=1 for Winograd compatibility
            padding=1,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Winograd-optimized convolution (handled by cuDNN backend).

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W).
        """
        x = self.conv(x)
        return x

batch_size = 1
in_channels = 16
out_channels = 32
height = 224
width = 224

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, True]