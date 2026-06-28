import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that computes element-wise sine function.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the sine of each element in the input tensor.
        Output has the same shape as input.
        Input values are interpreted as radians.
        
        output = sin(input)
    """
    
    def __init__(self):
        """
        Initialize the Sin model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input: Input tensor (values in radians)
        
        Returns:
            Tensor with element-wise sine, values in range [-1, 1]
        """
        return torch.sin(input)


# ======== Example input configuration ========

dim1 = 16
dim2 = 32
dim3 = 32

def get_inputs():
    input = torch.randn(dim1, dim2, dim3)
    return [input]

def get_init_inputs():
    return []