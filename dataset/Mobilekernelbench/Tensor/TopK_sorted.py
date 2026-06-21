import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs Top-K operation with sorted output.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Selects the k largest elements and ensures they are sorted.
        Returns sorted values in descending order.
    """
    
    def __init__(self, k: int = 10, dim: int = -1, sorted: bool = True):
        super(Model, self).__init__()
        self.k = k
        self.dim = dim
        self.sorted = sorted
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, dim]
        
        Returns:
            Sorted Top-K values tensor with shape [batch_size, k]
        """
        values, _ = torch.topk(x, self.k, dim=self.dim, sorted=self.sorted)
        return values


# ======== Example input configuration ========

batch_size = 32
dim = 256
k = 10
topk_dim = -1
sorted_output = True

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [k, topk_dim, sorted_output]