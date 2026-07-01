import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that computes the determinant with multiple batch dimensions.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the determinant of matrices with multiple leading batch dimensions.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch1, batch2, n, n]
        
        Returns:
            Output tensor with shape [batch1, batch2] containing determinants
        """
        return torch.linalg.det(x)


# ======== Example input configuration ========

batch_size1 = 4
batch_size2 = 128
matrix_size = 64

def get_inputs():
    x = torch.randn(batch_size1, batch_size2, matrix_size, matrix_size)
    return [x]

def get_init_inputs():
    return []