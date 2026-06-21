import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    """
    ReduceLogSumExp over multiple axes
    """
    def __init__(self, axes: Tuple[int, ...], keepdims: int = 0):
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = keepdims
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        LogSumExp over multiple specified dimensions
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor with LogSumExp over multiple dimensions
        """
        return torch.logsumexp(data, dim=self.axes, keepdim=bool(self.keepdims))

axes = (0, 2)
keepdims = 0

def get_inputs():
    data = torch.randn(2, 3, 4)
    return [data]

def get_init_inputs():
    return [axes, keepdims]