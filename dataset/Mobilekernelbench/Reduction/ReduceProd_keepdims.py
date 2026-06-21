import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceProd with keeping dimensions
    """
    def __init__(self, dim: int, keepdim: bool = True):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Product over specified dimension while keeping it as size 1
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with specified dimension kept as 1 after product
        """
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim)

batch_size = 16
input_shape = (64, 64, 32)
dim = 1

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape) * 0.1 + 0.9]

def get_init_inputs():
    return [dim]