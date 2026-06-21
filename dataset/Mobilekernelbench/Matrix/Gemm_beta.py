import torch
import torch.nn as nn
from typing import List


class Model(nn.Module):
    """
    A model that performs GEMM (General Matrix Multiplication) operation.
    
    Number of inputs: 3
    Implementation type: direct
    
    Semantics:
        Y = alpha * (A @ B) + beta * C
        Supports optional transpose of A and B matrices.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, transA: int = 0, transB: int = 0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            a: First input tensor with shape [M, K] or [K, M] if transA=1
            b: Second input tensor with shape [K, N] or [N, K] if transB=1
            c: Bias tensor that broadcasts to output shape
        
        Returns:
            Output tensor with shape [M, N]
        """
        a_mat = a.T if self.transA == 1 else a
        b_mat = b.T if self.transB == 1 else b
        return self.alpha * torch.mm(a_mat, b_mat) + self.beta * c


# ======== Example input configuration ========

M = 8
K = 64
N = 64
alpha = 1
beta = 0.5
transA = 0
transB = 0

def get_inputs():
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.randn(1, N)
    return [a, b, c]

def get_init_inputs():
    return [alpha, beta, transA, transB]