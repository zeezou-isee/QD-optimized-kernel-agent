import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise exponential.

    Semantics:
        y = exp(x)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


# ======== Example input configuration ========

batch_size = 16
input_shape = (512, 512)

def get_inputs():
    x = 10.0 * torch.rand(batch_size, *input_shape) - 5.0
    return [x]

def get_init_inputs():
    return []
