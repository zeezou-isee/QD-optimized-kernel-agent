import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies Layer Normalization without learnable parameters.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies Layer Normalization without affine transformation.
        Only normalizes to zero mean and unit variance.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after layer normalization (no affine)
        """
        return self.layer_norm(x)


# ======== Example input configuration ========

batch_size = 1
seq_len = 256
hidden_dim = 512

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return [x]

def get_init_inputs():
    return [hidden_dim]