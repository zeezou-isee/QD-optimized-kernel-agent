import torch
import torch.nn as nn
from typing import List, Optional


class Model(nn.Module):
    """
    A model that computes the log sum of exponentials along specified axes.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Computes log(sum(exp(x))) along the provided axes.
        This is numerically stable implementation of the operation.
        axes: Fixed reduction dimensions specified at initialization.
        keepdims: Whether to keep reduced dimensions (1) or not (0).
    """
    
    def __init__(self, axes: Optional[List[int]] = None, keepdims: int = 1):
        """
        Initialize the ReduceLogSumExp model.
        
        Args:
            axes: List of axes to reduce along (None means reduce all)
            keepdims: Whether to keep reduced dimensions (0 or 1)
        """
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: Input tensor to reduce
        
        Returns:
            Reduced tensor with log-sum-exp applied
        """
        if self.axes is not None:
            # Reduce along specified axes
            dim = tuple(self.axes)
            return torch.logsumexp(data, dim=dim, keepdim=self.keepdims)
        else:
            # Reduce over all dimensions
            result = torch.logsumexp(data.flatten(), dim=0)
            if self.keepdims:
                # Reshape to maintain original rank with all dims = 1
                return result.view([1] * data.ndim)
            else:
                return result


# ======== Example input configuration ========

dim1 = 2
dim2 = 3
dim3 = 4
reduce_axes = [0, 2]  # Axes to reduce along
keepdims = 1

def get_inputs():
    data = torch.randn(dim1, dim2, dim3)
    return [data]

def get_init_inputs():
    return [reduce_axes, keepdims]