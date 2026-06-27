import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that sums all elements using Einsum.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Sums all elements in the tensor
        Equation: "ij->"
    """
    
    def __init__(self, equation: str = "ij->"):
        super(Model, self).__init__()
        self.equation = equation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [M, N]
        
        Returns:
            Scalar tensor containing sum of all elements
        """
        return torch.einsum(self.equation, x)


# ======== Example input configuration ========

M = 64
N = 256
equation = "ij->"

def get_inputs():
    x = torch.randn(M, N)
    return [x]

def get_init_inputs():
    return [equation]