import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that applies 3D Instance Normalization.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes input over spatial dimensions (D, H, W) for each channel independently.
        Commonly used for 3D images (e.g., medical imaging, video).
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        Args:
            num_features: Number of channels (C) in the input
            eps: Small value added to variance for numerical stability
            affine: If True, learnable affine parameters (gamma, beta)
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm3d(
            num_features=num_features,
            eps=eps,
            affine=affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, num_features, depth, height, width]
        
        Returns:
            Output tensor with same shape as input
        """
        return self.inorm(x)


# ======== Example input configuration ========

batch_size = 1
num_features = 1
depth = 8
height = 64
width = 64
eps = 1e-5
affine = False

def get_inputs():
    x = torch.randn(batch_size, num_features, depth, height, width)
    return [x]

def get_init_inputs():
    return [num_features, eps, affine]