import torch
import torch.nn as nn
from typing import List, Tuple, Union

class Model(nn.Module):
    """
    ReduceSum over multiple axes
    """
    def __init__(self, dims: Union[List[int], Tuple[int, ...]], keepdim: bool = False):
        super(Model, self).__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum over multiple specified dimensions
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with summation over multiple dimensions
        """
        return torch.sum(x, dim=self.dims, keepdim=self.keepdim)

batch_size = 1
input_shape = (128, 64, 64)
dims = (2, 3)  # Sum over both spatial dimensions
keepdim = False

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dims, keepdim]