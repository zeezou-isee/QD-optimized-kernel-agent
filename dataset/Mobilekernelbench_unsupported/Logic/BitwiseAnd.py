import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies bitwise AND operation to two input tensors.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes the bitwise AND of two integer tensors element-wise.
        Supports NumPy-style broadcasting.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: First input tensor (integer type)
            y: Second input tensor (integer type)
        
        Returns:
            Output tensor after bitwise AND operation
        """
        return torch.bitwise_and(x, y)


# ======== Example input configuration ========

dim0 = 8
dim1 = 32
dim2 = 64

def get_inputs():
    x = torch.randint(0, 256, (dim0, dim1, dim2), dtype=torch.int32)
    y = torch.randint(0, 256, (dim0, dim1, dim2), dtype=torch.int32)
    return [x, y]

def get_init_inputs():
    return []