import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that returns the lower triangular part of a matrix.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Returns the lower triangular part of the input matrix.
        Elements on and below the diagonal (offset=0) are kept.
        Elements above the diagonal are set to zero.
    """
    
    def __init__(self, upper: int = 0):
        """
        Initialize the Trilu model.
        
        Args:
            upper: 1 for upper triangle, 0 for lower triangle
        """
        super(Model, self).__init__()
        self.upper = upper
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [rows, cols]
        
        Returns:
            Output tensor with triangular part extracted
        """
        diagonal = 0
        if self.upper:
            return torch.triu(x, diagonal=diagonal)
        else:
            return torch.tril(x, diagonal=diagonal)


# ======== Example input configuration ========

rows = 32
cols = 64
upper = 0

def get_inputs():
    x = torch.randint(0, 10, (rows, cols), dtype=torch.int64)
    return [x]

def get_init_inputs():
    return [upper]