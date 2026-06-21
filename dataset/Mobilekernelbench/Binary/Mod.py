import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise modulo operation.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes element-wise modulo C = A mod B with broadcasting support.
        - If fmod=0: Uses sign of divisor (Python/NumPy style, torch.remainder)
        - If fmod=1: Uses sign of dividend (C/C++ style, torch.fmod)
        
        For fmod=0: C = A - B * floor(A/B)
        For fmod=1: C = A - B * trunc(A/B)
    """
    
    def __init__(self, fmod: int = 0):
        """
        Initialize the Mod model.
        
        Args:
            fmod: Modulo mode (0 for remainder, 1 for fmod)
        """
        super(Model, self).__init__()
        self.fmod = fmod
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: Dividend tensor
            b: Divisor tensor
        
        Returns:
            Element-wise modulo result with broadcasting
        """
        if self.fmod == 1:
            return torch.fmod(a, b)
        else:
            return torch.remainder(a, b)


# ======== Example input configuration ========

dim1 = 8
dim2 = 256
dim3 = 128
fmod = 0  # 0 for remainder (Python style), 1 for fmod (C style)

def get_inputs():
    a = torch.randn(dim1, dim2, dim3)
    b = torch.randn(dim1, dim2, dim3)
    return [a, b]

def get_init_inputs():
    return [fmod]