import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMax with negative values in input
    """
    def __init__(self, dim: int, keepdim: bool = False):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maximum over specified dimension with negative values
        
        Args:
            x: Input tensor (may contain negative values)
            
        Returns:
            Tensor with maximum values
        """
        return torch.max(x, dim=self.dim, keepdim=self.keepdim).values

batch_size = 16
input_shape = (128, 128, 64)
dim = 1
keepdim = False

def get_inputs() -> list:
    # Generate values in range [-1, 1]
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim, keepdim]