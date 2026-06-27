import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs Top-K operation on 3D tensors.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Selects the k largest elements from 3D tensor along specified dimension.
        Useful for sequence or batched operations.
    """
    
    def __init__(self, k: int = 10, dim: int = -1):
        super(Model, self).__init__()
        self.k = k
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, dim]
        
        Returns:
            Top-K values tensor with shape [batch_size, seq_len, k]
        """
        values, _ = torch.topk(x, self.k, dim=self.dim)
        return values


# ======== Example input configuration ========

batch_size = 16
seq_len = 64
dim = 128
k = 10
topk_dim = -1

def get_inputs():
    x = torch.randn(batch_size, seq_len, dim)
    return [x]

def get_init_inputs():
    return [k, topk_dim]