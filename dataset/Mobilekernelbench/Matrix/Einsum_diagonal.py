import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that extracts diagonal elements using Einsum.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Extracts the diagonal of a square matrix
        Equation: "ii->i"
    """
    
    def __init__(self, equation: str = "ii->i"):
        super(Model, self).__init__()
        self.equation = equation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [N, N]
        
        Returns:
            Output tensor with shape [N] containing diagonal elements
        """
        return torch.einsum(self.equation, x)


# ======== Example input configuration ========

N = 1024
equation = "ii->i"

def get_inputs():
    x = torch.randn(N, N)
    return [x]

def get_init_inputs():
    return [equation]