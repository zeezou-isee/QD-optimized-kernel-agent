import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs matrix multiplication on two input tensors.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs batched matrix multiplication: output = a @ b
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor with shape [batch_size, ..., n, m]
            b: Second input tensor with shape [batch_size, ..., m, p]
        
        Returns:
            Output tensor with shape [batch_size, ..., n, p]
        """
        return torch.matmul(a, b)


# ======== Example input configuration ========

batch_dim1 = 8
batch_dim2 = 1
n = 16
m = 32
batch_dim3 = 1
batch_dim4 = 8
p = 32

def get_inputs():
    a = torch.randn(batch_dim1, batch_dim2, n, m)
    b = torch.randn(batch_dim3, batch_dim4, m, p)
    return [a, b]

def get_init_inputs():
    return []