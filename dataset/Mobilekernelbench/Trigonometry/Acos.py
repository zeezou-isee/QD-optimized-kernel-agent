import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise arccosine (Acos) of the input tensor.

    Semantics:
        y = acos(x)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.acos(x)


# ======== Example input configuration ========

batch_size = 16
input_shape = (128, 128)

def get_inputs():
    x = 2.0 * torch.rand(batch_size, *input_shape) - 1.0
    return [x]

def get_init_inputs():
    return []
