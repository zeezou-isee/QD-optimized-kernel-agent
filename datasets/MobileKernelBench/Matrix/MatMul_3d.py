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
        Input shapes: a = [batch, m, k], b = [batch, k, n]
        Output shape: [batch, m, n]
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor with shape [batch, m, k]
            b: Second input tensor with shape [batch, k, n]
        
        Returns:
            Output tensor with shape [batch, m, n] after matrix multiplication
        """
        return torch.matmul(a, b)


# ======== Example input configuration ========

batch = 8
m = 16
k = 32
n = 64

def get_inputs():
    a = torch.randn(batch, m, k)
    b = torch.randn(batch, k, n)
    return [a, b]

def get_init_inputs():
    return []