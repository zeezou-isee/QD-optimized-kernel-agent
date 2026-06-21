import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that applies 1D Instance Normalization.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes input over temporal/sequence dimension for each channel independently.
        Commonly used for sequence data or 1D signals.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        Args:
            num_features: Number of channels (C) in the input
            eps: Small value added to variance for numerical stability
            affine: If True, learnable affine parameters (gamma, beta)
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm1d(
            num_features=num_features,
            eps=eps,
            affine=affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, num_features, length]
        
        Returns:
            Output tensor with same shape as input
        """
        return self.inorm(x)


# ======== Example input configuration ========

batch_size = 1
num_features = 32
length = 128
eps = 1e-5
affine = False

def get_inputs():
    x = torch.randn(batch_size, num_features, length)
    return [x]

def get_init_inputs():
    return [num_features, eps, affine]