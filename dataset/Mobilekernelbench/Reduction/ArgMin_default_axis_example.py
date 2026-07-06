import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: Tensor
        
        Returns:
            Output tensor
        """
        return torch.argmin(data)

batch_size = 16
dim1 = 128
dim2 = 127

def get_inputs() -> List[torch.Tensor]:
    """
    Generate example inputs based on real ONNX test data
    
    Returns:
        List of input tensors
    """
    return [
        torch.randn(batch_size, dim1, dim2),  # data
    ]

def get_init_inputs() -> List:
    """Get initialization inputs (if any)"""
    return []
