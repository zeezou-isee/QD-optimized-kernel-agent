import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise division (Div) with broadcasting.

    Semantics:
        y = a / b
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.div(a, b)


# ======== Example input configuration ========
# a: (batch, C, H, W)
# b: (1, C, 1, 1)
batch_size = 4
C, H, W = 3, 64, 64

def get_inputs():
    a = torch.rand(batch_size, C, H, W)  # numerator
    b = 0.5 + torch.rand(1, C, 1, 1)
    return [a, b]

def get_init_inputs():
    return []
