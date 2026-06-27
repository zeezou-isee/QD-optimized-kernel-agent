import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs matrix multiplication on two input tensors.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs matrix multiplication: output = a @ b
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor with shape [n, m]
            b: Second input tensor with shape [m, p]
        
        Returns:
            Output tensor with shape [n, p]
        """
        return torch.matmul(a, b)


# ======== Example input configuration ========

n = 128
m = 128
p = 128

def get_inputs():
    a = torch.randn(n, m)
    b = torch.randn(m, p)
    return [a, b]

def get_init_inputs():
    return []