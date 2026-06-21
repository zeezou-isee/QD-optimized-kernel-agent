import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies 2D Global Average Pooling to the input tensor.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global average pooling across spatial dimensions (H, W).
        Output shape: (N, C, 1, 1)
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, C, H, W]
        
        Returns:
            Output tensor with shape [N, C, 1, 1]
        """
        return F.adaptive_avg_pool2d(x, output_size=1)


# ======== Example input configuration ========

batch_size = 1
channels = 3
height = 128
width = 128

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return []