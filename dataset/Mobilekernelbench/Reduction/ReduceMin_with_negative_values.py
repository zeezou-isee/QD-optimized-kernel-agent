import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMin with negative values in input
    """
    def __init__(self, dim: int, keepdim: bool = False):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Minimum over specified dimension with negative values
        
        Args:
            x: Input tensor (may contain negative values)
            
        Returns:
            Tensor with minimum values (likely negative)
        """
        return torch.min(x, dim=self.dim, keepdim=self.keepdim).values

batch_size = 16
input_shape = (128, 128, 64)
dim = 1
keepdim = False

def get_inputs() -> list:
    # Generate values in range [-1, 1]
    return [torch.rand(batch_size, *input_shape) * 2 - 1]

def get_init_inputs():
    return [dim, keepdim]