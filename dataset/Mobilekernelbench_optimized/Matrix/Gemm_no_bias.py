import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs GEMM (General Matrix Multiplication) operation without bias.
    
    Number of inputs: 2
    Implementation type: direct
    
    Semantics:
        Y = alpha * (A @ B)
        Matrix multiplication with optional transpose and scaling, no bias term.
    """
    
    def __init__(self, alpha: float = 1.0, transA: int = 0, transB: int = 0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.transA = transA
        self.transB = transB
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor with shape [M, K] or [K, M] if transA=1
            b: Second input tensor with shape [K, N] or [N, K] if transB=1
        
        Returns:
            Output tensor with shape [M, N]
        """
        a_mat = a.T if self.transA == 1 else a
        b_mat = b.T if self.transB == 1 else b
        return self.alpha * torch.mm(a_mat, b_mat)


# ======== Example input configuration ========

M = 16
K = 128
N = 32
alpha = 1.0
transA = 0
transB = 0

def get_inputs():
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    return [a, b]

def get_init_inputs():
    return [alpha, transA, transB]