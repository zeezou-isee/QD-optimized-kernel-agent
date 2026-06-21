import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that performs global max pooling operation.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies global max pooling across all spatial dimensions.
        Each channel is reduced to a single value by taking the maximum.
        
        For 2D input (N, C, H, W) -> output (N, C, 1, 1)
        For 3D input (N, C, D, H, W) -> output (N, C, 1, 1, 1)
        
        Y[n, c] = max(X[n, c, :, :, ...]) over all spatial dimensions
    """
    
    def __init__(self):
        """
        Initialize the GlobalMaxPool model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        
        Returns:
            Output tensor of shape (N, C, 1, 1) for 2D or (N, C, 1, 1, 1) for 3D
        
        Raises:
            ValueError: If input dimension is not 4D or 5D
        """
        if x.dim() == 4:
            # 2D Global Max Pooling: (N, C, H, W) -> (N, C, 1, 1)
            return F.adaptive_max_pool2d(x, output_size=1)
        elif x.dim() == 5:
            # 3D Global Max Pooling: (N, C, D, H, W) -> (N, C, 1, 1, 1)
            return F.adaptive_max_pool3d(x, output_size=1)
        else:
            raise ValueError(
                f"GlobalMaxPool expects 4D or 5D input, "
                f"but got {x.dim()}D input with shape {x.shape}"
            )


# ======== Example input configuration ========

batch_size = 1
channels = 3
height = 128
width = 64

def get_inputs():
    torch.manual_seed(42)
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return []