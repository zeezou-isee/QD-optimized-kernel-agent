import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that performs 3D global max pooling for video or medical imaging.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global max pooling across all spatial and temporal dimensions.
        Used for video classification, action recognition, or 3D medical imaging.
        
        Input: (N, C, D, H, W) -> Output: (N, C, 1, 1, 1)
        Y[n, c, 1, 1, 1] = max(X[n, c, d, h, w]) for all d, h, w
    """
    
    def __init__(self):
        """
        Initialize the 3D GlobalMaxPool model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C, D, H, W)
        
        Returns:
            Output tensor of shape (N, C, 1, 1, 1)
        """
        return F.adaptive_max_pool3d(x, output_size=1)


# ======== Example input configuration ========

batch_size = 1
channels = 256
depth = 16
height = 8
width = 8

def get_inputs():
    torch.manual_seed(42)
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return []