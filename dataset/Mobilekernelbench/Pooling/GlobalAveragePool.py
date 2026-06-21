import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies Global Average Pooling to the input tensor.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global average pooling across all spatial dimensions.
        Each channel is reduced to a single value by computing the mean
        of all spatial locations.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, C, H, W] for 2D or [N, C, D, H, W] for 3D
        
        Returns:
            Output tensor with shape [N, C, 1, 1] for 2D or [N, C, 1, 1, 1] for 3D
        """
        if x.dim() == 4:
            # 2D Global Average Pooling: (N, C, H, W) -> (N, C, 1, 1)
            return F.adaptive_avg_pool2d(x, output_size=1)
        elif x.dim() == 5:
            # 3D Global Average Pooling: (N, C, D, H, W) -> (N, C, 1, 1, 1)
            return F.adaptive_avg_pool3d(x, output_size=1)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


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