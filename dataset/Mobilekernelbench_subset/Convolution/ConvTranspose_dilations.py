import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class Model(nn.Module):
    """
    A model that applies 2D Transposed Convolution to the input tensor.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Applies transposed convolution operation with specified stride, padding, dilation, groups, and output_padding.
    """
    
    def __init__(self, strides: Tuple[int, int] = (1, 1),
                 pads: Optional[Tuple[int, int, int, int]] = None,
                 dilations: Tuple[int, int] = (2, 2),
                 group: int = 1,
                 output_padding: Tuple[int, int] = (0, 0)):
        super(Model, self).__init__()
        self.strides = strides
        self.pads = pads
        self.dilations = dilations
        self.group = group
        self.output_padding = output_padding
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            X: Input tensor with shape [batch_size, in_channels, height, width]
            W: Weight tensor with shape [in_channels, out_channels/groups, kernel_h, kernel_w]
        
        Returns:
            Output tensor after transposed convolution
        """
        # Convert pads to PyTorch padding format if provided
        # ONNX uses [top, left, bottom, right], PyTorch uses (left, right, top, bottom)
        if self.pads is not None:
            X = F.pad(X, (self.pads[1], self.pads[3], self.pads[0], self.pads[2]))
            padding = 0
        else:
            padding = 0
        
        return F.conv_transpose2d(X, W, bias=None, stride=self.strides, 
                                 padding=padding, output_padding=self.output_padding,
                                 groups=self.group, dilation=self.dilations)


# ======== Example input configuration ========

batch_size = 1
in_channels = 16
height = 32
width = 32
out_channels = 4
kernel_h = 32
kernel_w = 32

def get_inputs():
    X = torch.randn(batch_size, in_channels, height, width)
    W = torch.randn(in_channels, out_channels, kernel_h, kernel_w)
    return [X, W]

def get_init_inputs():
    return [(1, 1), None, (2, 2), 1, (0, 0)]  # [strides, pads, dilations, group, output_padding]