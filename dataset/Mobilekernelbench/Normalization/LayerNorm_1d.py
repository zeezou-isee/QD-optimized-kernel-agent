import torch
import torch.nn as nn
from typing import List, Tuple


class Model(nn.Module):
    """
    A model that applies Layer Normalization over a single dimension.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes input over the last dimension.
        Commonly used in transformers and sequence models.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
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
features = 512
eps = 1e-5
elementwise_affine = True

def get_inputs():
    x = torch.randn(batch_size, seq_len, features)
    return [x]

def get_init_inputs():
    return [features, eps, elementwise_affine]