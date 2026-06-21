import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies 3D Global Average Pooling to the input tensor.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global average pooling across spatial dimensions (D, H, W).
        Output shape: (N, C, 1, 1, 1)
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, C, D, H, W]
        
        Returns:
            Output tensor with shape [N, C, 1, 1, 1]
        """
        return F.adaptive_avg_pool3d(x, output_size=1)


# ======== Example input configuration ========

batch_size = 1
channels = 3
depth = 16
height = 64
width = 64

def get_inputs():
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return []