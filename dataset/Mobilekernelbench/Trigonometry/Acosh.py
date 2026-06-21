import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies the inverse hyperbolic cosine function element-wise.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Calculates the hyperbolic arccosine (acosh) of the input tensor element-wise.
        Input values must be in the range [1, inf).
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with values >= 1.0
        
        Returns:
            Output tensor after applying acosh function
        """
        return torch.acosh(x)


# ======== Example input configuration ========

dim0 = 8
dim1 = 64
dim2 = 64

def get_inputs():
    x = torch.rand(dim0, dim1, dim2) + 1.0  # Range: [1.0, 2.0)
    return [x]

def get_init_inputs():
    return []