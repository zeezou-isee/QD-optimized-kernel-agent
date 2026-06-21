import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceProd on 1D tensor
    """
    def __init__(self, dim: int = 0, keepdim: bool = False):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Product over 1D tensor
        
        Args:
            x: 1D input tensor
            
        Returns:
            Scalar or 1D tensor with single element
        """
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim).unsqueeze(0)

vector_size = 4096
dim = 0
keepdim = False

def get_inputs() -> list:
    return [torch.rand(vector_size) * 0.5 + 0.5]

def get_init_inputs():
    return [dim, keepdim]