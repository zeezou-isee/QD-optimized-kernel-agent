import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies logical XOR operation to two input tensors.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Computes the logical XOR of two boolean tensors element-wise.
        Supports NumPy-style broadcasting.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: First input tensor (boolean type)
            y: Second input tensor (boolean type)
        
        Returns:
            Output tensor after logical XOR operation
        """
        return torch.logical_xor(x, y)


# ======== Example input configuration ========

dim0 = 64
dim1 = 512

def get_inputs():
    x = torch.randint(0, 2, (dim0, dim1)).bool()
    y = torch.randint(0, 2, (dim0, dim1)).bool()
    return [x, y]

def get_init_inputs():
    return []