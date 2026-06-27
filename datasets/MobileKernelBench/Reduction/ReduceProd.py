import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim)

batch_size = 8
input_shape = (64, 32, 32)    
dim = 1
keepdim = False

# ======== Example input configuration ========
def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim, keepdim]