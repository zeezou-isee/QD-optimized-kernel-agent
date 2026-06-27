import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise error function (Erf).

    Semantics:
        y = erf(x)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


# ======== Example input configuration ========

batch_size = 1
input_shape = (64, 64)

def get_inputs():
    x = 6.0 * torch.rand(batch_size, *input_shape) - 3.0
    return [x]

def get_init_inputs():
    return []
