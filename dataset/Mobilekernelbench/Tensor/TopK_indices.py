import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs Top-K operation and returns only indices.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Selects the k largest elements along the specified dimension.
        Returns only the indices, not the values.
    """
    
    def __init__(self, k: int = 10, dim: int = -1):
        super(Model, self).__init__()
        self.k = k
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, dim]
        
        Returns:
            Top-K indices tensor with shape [batch_size, k]
        """
        _, indices = torch.topk(x, self.k, dim=self.dim)
        return indices


# ======== Example input configuration ========

batch_size = 32
dim = 256
k = 10
topk_dim = -1

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [k, topk_dim]