import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Basic GEMM (General Matrix Multiplication)
    Implements: Y = alpha * A @ B (+ beta * C)
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0, beta: float = 1.0, with_bias: bool = True):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_bias = with_bias
        self.linear = nn.Linear(in_features, out_features, bias=with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Y = alpha * (A @ W^T + b * beta)
        y = self.alpha * self.linear(x)
        return y

# ------------------- Hyperparameters -------------------
batch_size = 32
in_features = 512
out_features = 256
alpha = 1.0
beta = 1.0
with_bias = True

# ------------------- Input definitions -----------------
def get_inputs():
    x = torch.rand(batch_size, in_features)
    return [x]

def get_init_inputs():
    return [in_features, out_features, alpha, beta, with_bias]
