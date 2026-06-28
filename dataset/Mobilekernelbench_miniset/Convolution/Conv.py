import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Model(nn.Module):
    """
    A model that performs convolution operation (1D/2D/3D).
    
    Number of inputs: 2 or 3 (X, W, optional B)
    Implementation type: functional
    
    Semantics:
        Applies convolution operation with configurable parameters.
        Automatically detects 1D/2D/3D convolution based on weight dimensions.
        Supports groups, padding, strides, and dilations.
        
        Output shape calculation:
        output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """
    
    def __init__(
        self,
        group: int = 1,
        pads: List[int] = None,
        kernel_shape: List[int] = None,
        strides: List[int] = None,
        auto_pad: str = 'NOTSET',
        dilations: List[int] = None
    ):
        """
        Initialize the Conv model.
        
        Args:
            group: Number of groups for grouped convolution
            pads: Padding values [pad_left, pad_right, pad_top, pad_bottom, ...]
            kernel_shape: Kernel dimensions (typically inferred from weight)
            strides: Stride values for each spatial dimension
            auto_pad: Auto padding mode ('NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID')
            dilations: Dilation values for each spatial dimension
        """
        super(Model, self).__init__()
        self.group = group
        self.pads = pads if pads is not None else []
        self.kernel_shape = kernel_shape if kernel_shape is not None else []
        self.strides = strides if strides is not None else []
        self.auto_pad = auto_pad
        self.dilations = dilations if dilations is not None else []
    
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C_in, D1, D2, ...) 
            w: Weight tensor of shape (C_out, C_in/group, K1, K2, ...)
            b: Optional bias tensor of shape (C_out,)
        
        Returns:
            Output tensor of shape (N, C_out, D1_out, D2_out, ...)
        """
        # Determine convolution dimensionality from weight shape
        conv_dim = w.ndim - 2  # Subtract (out_channels, in_channels/group)
        
        # Set default values
        strides = self.strides if self.strides else [1] * conv_dim
        dilations = self.dilations if self.dilations else [1] * conv_dim
        pads = self.pads if self.pads else [0] * (conv_dim * 2)
        
        # Convert ONNX-style padding (begin_1, begin_2, ..., end_1, end_2, ...)
        # to PyTorch-style (pad_1, pad_2, ...)
        # For simplicity, use symmetric padding (first half of pads array)
        if len(pads) >= conv_dim:
            padding = tuple(int(p) for p in pads[:conv_dim])
        else:
            padding = 0
        
        # Select convolution function based on dimensionality
        if conv_dim == 1:
            return F.conv1d(
                x, w, b,
                stride=tuple(int(s) for s in strides),
                padding=padding,
                dilation=tuple(int(d) for d in dilations),
                groups=self.group
            )
        elif conv_dim == 2:
            return F.conv2d(
                x, w, b,
                stride=tuple(int(s) for s in strides),
                padding=padding,
                dilation=tuple(int(d) for d in dilations),
                groups=self.group
            )
        elif conv_dim == 3:
            return F.conv3d(
                x, w, b,
                stride=tuple(int(s) for s in strides),
                padding=padding,
                dilation=tuple(int(d) for d in dilations),
                groups=self.group
            )
        else:
            raise ValueError(f"Unsupported convolution dimension: {conv_dim}")


# ======== Example input configuration ========

# Conv2D example
batch_size = 1
in_channels = 3
out_channels = 16
height = 32
width = 32
kernel_size = 3

# Convolution parameters
groups = 1
strides = [1, 1]
pads = [1, 1, 1, 1]  # [left, right, top, bottom]
dilations = [1, 1]
auto_pad = 'NOTSET'

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
    b = torch.randn(out_channels)
    return [x, w, b]

def get_init_inputs():
    return [groups, pads, [], strides, auto_pad, dilations]