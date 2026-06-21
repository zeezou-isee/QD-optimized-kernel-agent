import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise power operation.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes element-wise power C = A^B with broadcasting support.
        For each element: C[i] = A[i] ** B[i]
        Supports NumPy-style broadcasting for tensors of different shapes.
        
        Note: For negative bases with non-integer exponents, result will be NaN.
    """
    
    def __init__(self):
        """
        Initialize the Pow model.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Base tensor
            y: Exponent tensor
        
        Returns:
            Element-wise power result with broadcasting
        """
        return torch.pow(x, y)


# ======== Example input configuration ========

dim1 = 8
dim2 = 256
dim3 = 256

def get_inputs():
    torch.manual_seed(42)
    x = torch.rand(dim1, dim2, dim3) * 10.999 + 0.001  # [0.001, 11.0)  
    y = torch.rand(dim1, dim2, dim3) * 1.999 + 0.001  # [0.001, 2.0)
    return [x, y]

def get_init_inputs():
    return []