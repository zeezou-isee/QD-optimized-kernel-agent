import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Top-K operation, selecting the k largest elements.
    """
    def __init__(self, k: int = 10):
        super(Model, self).__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Top-K operation to the input tensor along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Top-K values tensor of shape (batch_size, k).
        """
        values, indices = torch.topk(x, self.k, dim=-1)
        return values

batch_size = 64
dim = 1024
K = 10

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [K]  # k value for Top-K operation