import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMean with keeping dimensions
    """
    def __init__(self, dim: int, keepdim: bool = True):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean over specified dimension while keeping it as size 1
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with specified dimension kept as 1 after averaging
        """
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

batch_size = 8
input_shape = (64, 256, 256)
dim = 1

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]