import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise multiplication using Einsum.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs element-wise multiplication of two matrices
        Equation: "ij,ij->ij"
    """
    
    def __init__(self, equation: str = "ij,ij->ij"):
        super(Model, self).__init__()
        self.equation = equation
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x1: First input tensor with shape [M, N]
            x2: Second input tensor with shape [M, N]
        
        Returns:
            Output tensor with shape [M, N]
        """
        return torch.einsum(self.equation, x1, x2)


# ======== Example input configuration ========

M = 256
N = 256
equation = "ij,ij->ij"

def get_inputs():
    x1 = torch.randn(M, N)
    x2 = torch.randn(M, N)
    return [x1, x2]

def get_init_inputs():
    return [equation]