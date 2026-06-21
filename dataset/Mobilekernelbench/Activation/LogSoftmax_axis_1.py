import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies LogSoftmax operation to the input tensor.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Applies log(softmax(x)) along the specified axis.
    """
    
    def __init__(self, axis: int = 1):
        super(Model, self).__init__()
        self.axis = axis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after applying log_softmax
        """
        return F.log_softmax(x, dim=self.axis)


# ======== Example input configuration ========

dim0 = 8
dim1 = 32
dim2 = 64

def get_inputs():
    x = torch.randn(dim0, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # [axis]