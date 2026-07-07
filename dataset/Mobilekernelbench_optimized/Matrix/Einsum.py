import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs matrix multiplication using Einsum.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs standard matrix multiplication: C = A @ B
        Equation: "ij,jk->ik"
    """
    
    def __init__(self, equation: str = "ij,jk->ik"):
        super(Model, self).__init__()
        self.equation = equation
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x1: First input tensor with shape [M, K]
            x2: Second input tensor with shape [K, N]
        
        Returns:
            Output tensor with shape [M, N]
        """
        return torch.einsum(self.equation, x1, x2)


# ======== Example input configuration ========

M = 32
K = 64
N = 128
equation = "ij,jk->ik"

def get_inputs():
    x1 = torch.randn(M, K)
    x2 = torch.randn(K, N)
    return [x1, x2]

def get_init_inputs():
    return [equation]