import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that computes element-wise hyperbolic tangent function.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the hyperbolic tangent of each element in the input tensor.
        Output has the same shape as input.
        Output is bounded: tanh(x) ∈ (-1, 1)
        
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                = (e^(2x) - 1) / (e^(2x) + 1)
        
    Note:
        Unlike tan(x), tanh(x) has no singularities and is smooth everywhere.
        Commonly used as an activation function in neural networks.
    """
    
    def __init__(self):
        """
        Initialize the Tanh model.
        
        Note: This operator has no learnable parameters.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of any shape
        
        Returns:
            Tensor with element-wise hyperbolic tangent, values in range (-1, 1)
        """
        return torch.tanh(x)


# ======== Example input configuration ========

dim1 = 8
dim2 = 128
dim3 = 256

def get_inputs():
    x = torch.randn(dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return []