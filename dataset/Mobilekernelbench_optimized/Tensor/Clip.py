import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)

batch_size = 16
input_shape = (32, 128, 256)
min_val = -0.5
max_val = 0.5

def get_inputs() -> list:
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [min_val, max_val]
