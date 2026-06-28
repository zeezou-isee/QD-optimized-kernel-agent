import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs transpose using Einsum.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Transposes a 2D matrix
        Equation: "ij->ji"
    """
    
    def __init__(self, equation: str = "ij->ji"):
        super(Model, self).__init__()
        self.equation = equation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [M, N]
        
        Returns:
            Output tensor with shape [N, M]
        """
        return torch.einsum(self.equation, x)


# ======== Example input configuration ========

M = 64
N = 256
equation = "ij->ji"

def get_inputs():
    x = torch.randn(M, N)
    return [x]

def get_init_inputs():
    return [equation]