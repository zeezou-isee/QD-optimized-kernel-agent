import torch
import torch.nn as nn
from typing import List, Tuple


class Model(nn.Module):
    """
    A model that applies Layer Normalization over last three dimensions.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes input over the last three dimensions.
        Useful for 4D tensors (batch, channels, height, width).
    """
    
    def __init__(self, normalized_shape: Tuple[int, int, int], eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        Args:
            normalized_shape: Tuple of (channels, height, width) to normalize
            eps: Small value added to variance for numerical stability
            elementwise_affine: If True, learnable affine parameters
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor with same shape as input
        """
        return self.ln(x)


# ======== Example input configuration ========

batch_size = 1
channels = 8
height = 32
width = 32
eps = 1e-5
elementwise_affine = True

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [(height, width), eps, elementwise_affine]