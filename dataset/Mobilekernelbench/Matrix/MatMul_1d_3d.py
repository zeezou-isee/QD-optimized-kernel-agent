import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs matrix multiplication on two input tensors.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs matrix multiplication with broadcasting: output = a @ b
        When a is 1D and b is 3D, a is treated as a row vector.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor (1D vector)
            b: Second input tensor (multi-dimensional)
        
        Returns:
            Output tensor after matrix multiplication
        """
        return torch.matmul(a, b)


# ======== Example input configuration ========

m = 16
batch_size = 32
p = 64

def get_inputs():
    a = torch.randn(m)
    b = torch.randn(batch_size, m, p)
    return [a, b]

def get_init_inputs():
    return []