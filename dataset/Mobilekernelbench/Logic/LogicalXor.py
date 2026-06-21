import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise logical XOR operation.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes element-wise logical XOR: C = A XOR B with broadcasting support.
        Returns boolean tensor where each element is True if exactly one of A or B is True.
        Non-zero values are treated as True, zero values as False.
        Supports NumPy-style broadcasting for tensors of different shapes.
    """
    
    def __init__(self):
        """
        Initialize the Xor model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor (numeric or boolean)
            b: Second input tensor (numeric or boolean)
        
        Returns:
            Boolean tensor with element-wise logical XOR result
        """
        return torch.logical_xor(a, b)


# ======== Example input configuration ========

dim1 = 8
dim2 = 128
dim3 = 128

def get_inputs():
    a = torch.randn(dim1, dim2, dim3)
    b = torch.randn(dim1, dim2, dim3)
    return [a, b]

def get_init_inputs():
    return []