import torch
import torch.nn as nn
from typing import List, Union, Tuple


class Model(nn.Module):
    """
    A model that performs 3D convolution operation.
    
    Number of inputs: 1 (or 3 with optional bias and weight)
    Implementation type: direct
    
    Semantics:
        Applies a 3D convolution over an input tensor with shape [N, C_in, D, H, W].
        The convolution kernel has shape [C_out, C_in/groups, kD, kH, kW].
        
        Output shape: [N, C_out, D_out, H_out, W_out]
        
        Where:
            D_out = floor((D + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
            H_out = floor((H + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
            W_out = floor((W + 2*padding[2] - dilation[2]*(kernel_size[2]-1) - 1) / stride[2] + 1)
    
    Applications:
        - Video processing and understanding
        - Medical image analysis (CT, MRI volumes)
        - 3D object recognition
        - Action recognition in videos
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        """
        Initialize the Conv3D model.
        
        Args:
            in_channels: Number of input channels (C_in)
            out_channels: Number of output channels (C_out)
            kernel_size: Size of the convolving kernel (depth, height, width)
            stride: Stride of the convolution (default: 1)
            padding: Zero-padding added to all three sides (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            groups: Number of blocked connections (default: 1)
            bias: If True, adds a learnable bias (default: True)
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular' (default: 'zeros')
        """
        super(Model, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, C_in, D, H, W]
               N: batch size
               C_in: number of input channels
               D: depth dimension
               H: height dimension
               W: width dimension
        
        Returns:
            Output tensor with shape [N, C_out, D_out, H_out, W_out]
        """
        return self.conv3d(x)


# ======== Example input configuration ========

batch_size = 1
in_channels = 32
out_channels = 64
depth = 8
height = 16
width = 16

kernel_size = (3, 3, 3)
stride = (1, 1, 1)
padding = (1, 1, 1)
dilation = (1, 1, 1)
groups = 1
bias = True

def get_inputs():
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias
    ]