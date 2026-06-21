import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs the Where operation with scalar values.
    
    Number of inputs: 3
    Implementation type: direct
    
    Semantics:
        Selects elements from x or y based on condition.
        Demonstrates usage with scalar values that broadcast to match condition shape.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            condition: Boolean tensor
            x: Scalar tensor (broadcasts to condition shape)
            y: Scalar tensor (broadcasts to condition shape)
        
        Returns:
            Output tensor with elements selected from x or y
        """
        return torch.where(condition, x, y)


# ======== Example input configuration ========

batch_size = 4
height = 128
width = 128

def get_inputs():
    condition = torch.rand(batch_size, height, width) > 0.5
    # Scalar values that will broadcast
    x = torch.tensor(1.0)
    y = torch.tensor(0.0)
    return [condition, x, y]

def get_init_inputs():
    return []