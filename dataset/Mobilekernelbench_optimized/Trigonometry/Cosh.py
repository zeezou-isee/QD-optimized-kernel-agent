import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies the hyperbolic cosine function element-wise.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Calculates the hyperbolic cosine (cosh) of the input tensor element-wise.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after applying cosh function
        """
        return torch.cosh(x)


# ======== Example input configuration ========

dim0 = 8
dim1 = 128
dim2 = 256

def get_inputs():
    x = torch.randn(dim0, dim1, dim2)
    return [x]

def get_init_inputs():
    return []