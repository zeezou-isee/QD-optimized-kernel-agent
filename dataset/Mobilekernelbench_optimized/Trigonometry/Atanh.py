import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies the inverse hyperbolic tangent function element-wise.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Calculates the hyperbolic arctangent (atanh) of the input tensor element-wise.
        Input values must be in the range (-1, 1).
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with values in the range (-1, 1)
        
        Returns:
            Output tensor after applying atanh function
        """
        return torch.atanh(x)


# ======== Example input configuration ========

dim0 = 8
dim1 = 64
dim2 = 64

def get_inputs():
    x = torch.rand(dim0, dim1, dim2) * 1.8 - 0.9  # Range: (-0.9, 0.9)
    return [x]

def get_init_inputs():
    return []