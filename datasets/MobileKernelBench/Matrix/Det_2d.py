import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that computes the determinant of 2x2 matrices.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the determinant of each 2x2 matrix in the input tensor.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [..., 2, 2]
        
        Returns:
            Output tensor with shape [...] containing determinants
        """
        return torch.linalg.det(x)


# ======== Example input configuration ========

batch_size = 64
matrix_size = 512

def get_inputs():
    x = torch.randn(batch_size, matrix_size, matrix_size)
    return [x]

def get_init_inputs():
    return []