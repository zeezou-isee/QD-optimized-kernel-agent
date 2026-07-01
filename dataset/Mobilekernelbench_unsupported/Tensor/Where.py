import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs the Where (Select) operation.
    
    Number of inputs: 3
    Implementation type: direct
    
    Semantics:
        Selects elements from x or y based on condition.
        When condition is True, yield x, otherwise yield y.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            condition: Boolean tensor indicating which elements to select
            x: First input tensor (selected when condition is True)
            y: Second input tensor (selected when condition is False)
        
        Returns:
            Output tensor with elements selected from x or y
        """
        return torch.where(condition, x, y)


# ======== Example input configuration ========

batch_size = 32
features = 512
dim1 = 512

def get_inputs():
    condition = torch.rand(batch_size, features, dim1) > 0.5
    x = torch.randn(batch_size, features, dim1)
    y = torch.randn(batch_size, features, dim1)
    return [condition, x, y]

def get_init_inputs():
    return []