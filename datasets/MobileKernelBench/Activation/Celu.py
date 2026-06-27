import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.celu(x, alpha=self.alpha)

batch_size = 128
dim = 1024
alpha = 1.0

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [alpha]
