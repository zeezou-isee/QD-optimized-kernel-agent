import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union


class Model(nn.Module):
    """
    Model that performs 2D Average Pooling with ceil mode.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        self.kernel_size = kernel_size  
        self.stride = stride if stride is not None else kernel_size  
        self.padding = padding  
    def forward(self, x) -> torch.Tensor:
        """
        Applies 2D Average Pooling with ceil mode to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, ceil_mode=True)

batch_size = 1
channels = 16
height = 64
width = 64
kernel_size = 11

def get_inputs() -> List[torch.Tensor]:
    """
    Generate example inputs based on real ONNX test data
    
    Returns:
        List of input tensors
    """
    return [
        torch.randn(batch_size, channels, height, width),  # x
    ]


def get_init_inputs() -> List:
    """Get initialization inputs (if any)"""
    return [kernel_size]