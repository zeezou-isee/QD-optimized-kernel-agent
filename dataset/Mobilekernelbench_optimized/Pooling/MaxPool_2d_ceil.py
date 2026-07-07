import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Model(nn.Module):
    """
    A model that applies 2D Max Pooling to the input tensor.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies max pooling operation with specified kernel size, stride, padding, and ceil mode.
    """
    
    def __init__(self, kernel_shape: Tuple[int, int] = (3, 3), 
                 strides: Tuple[int, int] = (2, 2), 
                 pads: Optional[Tuple[int, int, int, int]] = None,
                 ceil_mode: int = 1):
        super(Model, self).__init__()
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.pads = pads
        self.ceil_mode = bool(ceil_mode)  # Convert to boolean for PyTorch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor after max pooling
        """
        # Convert pads to PyTorch padding format if provided
        # ONNX uses [top, left, bottom, right], PyTorch uses (left, right, top, bottom)
        if self.pads is not None:
            x = F.pad(x, (self.pads[1], self.pads[3], self.pads[0], self.pads[2]))
            padding = 0
        else:
            padding = 0
        
        return F.max_pool2d(x, kernel_size=self.kernel_shape, 
                           stride=self.strides, 
                           padding=padding,
                           ceil_mode=self.ceil_mode)


# ======== Example input configuration ========

batch_size = 1
channels = 3
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [(3, 3), (2, 2), None, 1]  # [kernel_shape, strides, pads, ceil_mode]