import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that computes element-wise tangent function.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the tangent of each element in the input tensor.
        Output has the same shape as input.
        Input values are interpreted as radians.
        
        output = tan(input) = sin(input) / cos(input)
        
    Note:
        Tangent has singularities at x = (2k+1)π/2 where k is an integer.
        At these points, tan(x) approaches ±∞.
    """
    
    def __init__(self):
        """
        Initialize the Tan model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input: Input tensor (values in radians)
        
        Returns:
            Tensor with element-wise tangent, values in range (-∞, +∞)
        """
        return torch.tan(input)


# ======== Example input configuration ========

dim1 = 16
dim2 = 256
dim3 = 256

def get_inputs():
    input = torch.randn(dim1, dim2, dim3)
    return [input]

def get_init_inputs():
    return []