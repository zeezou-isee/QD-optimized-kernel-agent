import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    A model that applies the Exponential Linear Unit (ELU) activation function.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        ELU(x) = x if x > 0
        ELU(x) = alpha * (exp(x) - 1) if x <= 0
        Applies element-wise activation with configurable alpha parameter.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the ELU model.
        
        Args:
            alpha: Scaling factor for negative values (default: 1.0)
        """
        super(Model, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor with ELU activation applied
        """
        return F.elu(x, alpha=self.alpha)


# ======== Example input configuration ========

dim1 = 16
dim2 = 64
dim3 = 128
alpha = 0.1

def get_inputs():
    x = torch.randn(dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [alpha]