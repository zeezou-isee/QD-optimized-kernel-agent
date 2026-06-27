import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise cosine (Cos) of the input tensor.

    Semantics:
        y = cos(x)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


# ======== Example input configuration ========

batch_size = 16
input_shape = (256, 256)

def get_inputs():
    x = (2 * torch.pi) * torch.rand(batch_size, *input_shape) - torch.pi
    return [x]

def get_init_inputs():
    return []
