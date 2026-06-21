import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that applies Layer Normalization with higher epsilon value.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes input with a larger epsilon for stability.
        Tests numerical stability with different epsilon values.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-3, 
                 elementwise_affine: bool = True):
        """
        Args:
            normalized_shape: Size of the dimension to normalize
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
            x: Input tensor with shape [batch_size, seq_len, features]
        
        Returns:
            Output tensor with same shape as input
        """
        return self.ln(x)


# ======== Example input configuration ========

batch_size = 1
seq_len = 128
features = 256
eps = 1e-3
elementwise_affine = True

def get_inputs():
    x = torch.randn(batch_size, seq_len, features)
    return [x]

def get_init_inputs():
    return [features, eps, elementwise_affine]