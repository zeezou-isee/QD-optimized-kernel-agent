import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs GEMM with transposed input.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Y = alpha * (A^T @ W^T + b)
        Input is transposed before matrix multiplication.
    """
    
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [in_features, batch_size]
        
        Returns:
            Output tensor with shape [batch_size, out_features]
        """
        # Transpose input before linear transformation
        x_t = x.t()
        return self.alpha * self.linear(x_t)


# ======== Example input configuration ========

batch_size = 32
in_features = 128
out_features = 256
alpha = 1.5

def get_inputs():
    x = torch.randn(in_features, batch_size)
    return [x]

def get_init_inputs():
    return [in_features, out_features, alpha]