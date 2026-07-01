import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs the Where operation with broadcasting.
    
    Number of inputs: 3
    Implementation type: direct
    
    Semantics:
        Selects elements from x or y based on condition.
        Demonstrates broadcasting where condition and inputs have different shapes.
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            condition: Boolean tensor with shape [batch_size, 1, 1]
            x: First input tensor with shape [batch_size, channels, features]
            y: Second input tensor with shape [batch_size, channels, features]
        
        Returns:
            Output tensor with elements selected from x or y
        """
        return torch.where(condition, x, y)


# ======== Example input configuration ========

batch_size = 8
channels = 64
features = 512

def get_inputs():
    # Condition broadcasts across channels and features
    condition = torch.rand(batch_size, 1, 1) > 0.5
    x = torch.randn(batch_size, channels, features)
    y = torch.randn(batch_size, channels, features)
    return [condition, x, y]

def get_init_inputs():
    return []