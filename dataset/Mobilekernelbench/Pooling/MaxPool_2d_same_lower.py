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
        Applies max pooling operation with specified kernel size, stride, and padding.
    """
    
    def __init__(self, kernel_shape: Tuple[int, int] = (2, 2), 
                 strides: Tuple[int, int] = (1, 1), 
                 pads: Optional[Tuple[int, int, int, int]] = None,
                 auto_pad: str = "SAME_LOWER"):
        super(Model, self).__init__()
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.pads = pads
        self.auto_pad = auto_pad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor after max pooling
        """
        # Handle auto_pad
        if self.auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            # Calculate padding for 'SAME' mode
            padding = self._calculate_same_padding(x.shape[2:])
        elif self.pads is not None:
            # ONNX uses [top, left, bottom, right], PyTorch uses (left, right, top, bottom)
            x = F.pad(x, (self.pads[1], self.pads[3], self.pads[0], self.pads[2]))
            padding = 0
        else:
            padding = 0
        
        return F.max_pool2d(x, kernel_size=self.kernel_shape, 
                           stride=self.strides, padding=padding)
    
    def _calculate_same_padding(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate padding for SAME mode"""
        padding = []
        for i in range(2):
            out_size = (input_shape[i] + self.strides[i] - 1) // self.strides[i]
            pad_total = (out_size - 1) * self.strides[i] + self.kernel_shape[i] - input_shape[i]
            # pad_total = torch.clamp(torch.as_tensor(pad_total), min=0)
            padding.append(pad_total // 2)
        return tuple(padding)


# ======== Example input configuration ========

batch_size = 1
channels = 3
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [(2, 2), (1, 1), None, "SAME_LOWER"]  # [kernel_shape, strides, pads, auto_pad]