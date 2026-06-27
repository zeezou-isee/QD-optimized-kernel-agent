import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_slopes):
        super().__init__()
        self.slope = nn.Parameter(torch.randn(num_slopes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.prelu(x, self.slope)

batch_size = 64
dim = 1024

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [dim]
