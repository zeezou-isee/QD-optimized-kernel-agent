import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceL2 operation
    """
    def __init__(self, dim: tuple = None, keepdim: bool = True):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        L2 norm reduction over specified dimensions
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after L2 reduction
        """
        if self.dim is not None:
            return torch.norm(x, dim=self.dim, keepdim=self.keepdim)
        else:
            return torch.norm(x)

batch_size = 2
input_shape = (3, 4)
dim = (1,)      # can be None or tuple of dims
keepdim = True

def get_inputs() -> list:
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim, keepdim]