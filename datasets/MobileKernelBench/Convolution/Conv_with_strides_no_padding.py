import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Model(nn.Module):
    """
    A model that applies 2D Convolution to the input tensor.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Applies 2D convolution operation with specified stride, padding, dilation, and groups.
    """
    
    def __init__(self, strides: Tuple[int, int] = (2, 2),
                 pads: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 dilations: Tuple[int, int] = (1, 1),
                 group: int = 1,
                 kernel_shape: Tuple[int, int] = (3, 3)):
        super(Model, self).__init__()
        self.strides = strides
        self.pads = pads
        self.dilations = dilations
        self.group = group
        self.kernel_shape = kernel_shape
    
    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, in_channels, height, width]
            W: Weight tensor with shape [out_channels, in_channels/groups, kernel_h, kernel_w]
        
        Returns:
            Output tensor after convolution
        """
        # Convert ONNX padding format [top, left, bottom, right] to PyTorch format (left, right)
        # PyTorch conv2d uses symmetric padding (left/right, top/bottom)
        padding = (self.pads[0], self.pads[1])
        
        return F.conv2d(x, W, bias=None, stride=self.strides, 
                       padding=padding, dilation=self.dilations, groups=self.group)


# ======== Example input configuration ========


batch_size = 1
in_channels = 8
height = 64
width = 64
out_channels = 4
kernel_h = 64
kernel_w = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    W = torch.randn(out_channels, in_channels, kernel_h, kernel_w)
    return [x, W]

def get_init_inputs():
    return [(2, 2), (0, 0, 0, 0), (1, 1), 1, (3, 3)]  # [strides, pads, dilations, group, kernel_shape]