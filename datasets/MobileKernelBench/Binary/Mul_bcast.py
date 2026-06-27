import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs element-wise multiplication with broadcasting.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Performs element-wise multiplication: output = x * y
        y is broadcasted to match x's last dimension.
        x[8, 16, 16] * y[16] -> output[8, 16, 16]
    """
    
    def __init__(self):
        """
        Initialize the Mul model.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: First input tensor with shape [batch, dim1, dim2]
            y: Second input tensor with shape [dim2]
        
        Returns:
            Output tensor with element-wise multiplication applied
        """
        return x * y


# ======== Example input configuration ========

batch_size = 8
dim1 = 64
dim2 = 64

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    y = torch.randn(dim2)  # Changed to dim2=16 to match the last dimension of x
    return [x, y]

def get_init_inputs():
    return []