import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise logical NOT operation.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes element-wise logical negation of the input tensor.
        Non-zero values are treated as True, zero values as False.
        Output is boolean tensor where each element is the negation of input.
        
        Y[i] = NOT X[i]
    """
    
    def __init__(self):
        """
        Initialize the Not model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of any numeric or boolean type
        
        Returns:
            Boolean tensor with element-wise logical negation
        """
        return torch.logical_not(x)


# ======== Example input configuration ========

dim1 = 8
dim2 = 64
dim3 = 64

def get_inputs():
    x = torch.randn(dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return []