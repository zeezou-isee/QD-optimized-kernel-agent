import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise logical AND between two boolean tensors.

    Semantics:
        y = logical_and(a, b)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_bool = a.bool()
        b_bool = b.bool()
        return torch.logical_and(a_bool, b_bool)


# ======== Example input configuration ========
batch_size = 4
H, W = 32, 32

def get_inputs():
    a = (torch.rand(batch_size, H, W) > 0.5)
    b = (torch.rand(batch_size, H, W) > 0.5)
    return [a, b]

def get_init_inputs():
    return []
