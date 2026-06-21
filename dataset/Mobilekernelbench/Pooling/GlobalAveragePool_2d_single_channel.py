import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies 2D Global Average Pooling to a single-channel input.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global average pooling across spatial dimensions (H, W).
        Tests with single channel input.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, 1, H, W]
        
        Returns:
            Output tensor with shape [N, 1, 1, 1]
        """
        return F.adaptive_avg_pool2d(x, output_size=1)


# ======== Example input configuration ========

batch_size = 1
channels = 3
height = 64
width = 64

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return []