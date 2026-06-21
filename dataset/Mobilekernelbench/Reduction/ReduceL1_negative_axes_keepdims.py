import torch
import torch.nn as nn
from typing import List, Optional


class Model(nn.Module):
    """
    A model that computes L1 norm (sum of absolute values) reduction.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes the L1 norm of the input tensor along specified axes.
        L1 norm is the sum of absolute values: ||x||₁ = Σ|xᵢ|
        
        Supports:
        - Negative axis indices (e.g., -1 for last dimension)
        - keepdims option to preserve reduced dimensions
    
    Args:
        axes: Axes to reduce over (can be negative), default -1
        keepdims: Whether to keep reduced dimensions (1/True or 0/False)
    """
    
    def __init__(self, axes: int = -1, keepdims: int = 1):
        """
        Initialize the ReduceL1 model.
        
        Args:
            axes: Axis to reduce over (default: -1, last dimension)
            keepdims: Keep reduced dimensions if 1/True (default: 1)
        """
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: Input tensor
        
        Returns:
            Tensor with L1 norm computed along specified axes
        """
        # Compute L1 norm (sum of absolute values)
        return torch.norm(data, p=1, dim=self.axes, keepdim=self.keepdims)


# ======== Example input configuration ========

dim1 = 8
dim2 = 64
dim3 = 64
default_axes = -1
default_keepdims = 1

def get_inputs():
    torch.manual_seed(42)
    data = torch.randn(dim1, dim2, dim3)
    return [data]

def get_init_inputs():
    return [default_axes, default_keepdims]