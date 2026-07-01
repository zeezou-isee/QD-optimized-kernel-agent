import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union


class Model(nn.Module):
    
    def __init__(self, dim: Optional[int] = 0, sorted: bool = True):
        super(Model, self).__init__()
        self.dim = dim  
        self.sorted = sorted  
    
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            X: Tensor
        
        Returns:
            Tuple of (unique values, inverse indices, counts)
        """
        return torch.unique(
            X, 
            sorted=self.sorted,
            return_inverse=True, 
            return_counts=True,
            dim=self.dim
        )

batch_size = 64
dim = 512

# ======== Example input configuration ========
def get_inputs() -> List[torch.Tensor]:
    """
    Generate example inputs based on real ONNX test data
    
    Returns:
        List of input tensors
    """
    return [
        torch.randn(batch_size, dim),  # X
    ]


def get_init_inputs() -> List:
    """Get initialization inputs (if any)"""
    return [0, True]  # [dim, sorted]