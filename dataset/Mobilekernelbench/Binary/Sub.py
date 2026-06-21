import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise subtraction.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes element-wise subtraction C = A - B with broadcasting support.
        Supports NumPy-style broadcasting for tensors of different shapes.
    """
    
    def __init__(self):
        """
        Initialize the Sub model.
        """
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: Minuend tensor (minuend)
            b: Subtrahend tensor (subtrahend)
        
        Returns:
            Element-wise subtraction result with broadcasting
        """
        return torch.sub(a, b)


# ======== Example input configuration ========

dim1 = 8
dim2 = 128
dim3 = 256

def get_inputs():
    a = torch.randn(dim1, dim2, dim3)
    b = torch.randn(dim1, dim2, dim3)
    return [a, b]

def get_init_inputs():
    return []