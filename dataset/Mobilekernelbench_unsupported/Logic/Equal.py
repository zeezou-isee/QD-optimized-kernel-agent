import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise equality comparison.

    Semantics:
        y = (a == b)
    Output:
        bool tensor, broadcast semantics like ONNX Equal.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.eq(a, b)


# ======== Example input configuration ========

batch_size = 4
C, H, W = 3, 64, 128

def get_inputs():
    # a: (B, C, H, W)
    a = torch.rand(batch_size, C, H, W)
    b = torch.rand(batch_size, C, H, W)
    return [a, b]

def get_init_inputs():
    return []
