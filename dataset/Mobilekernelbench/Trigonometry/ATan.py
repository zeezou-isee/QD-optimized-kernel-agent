import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that applies element-wise arctangent (ATan) to the input tensor.

    Semantics:
        y = atan(x)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)


# ======== Example input configuration ========

batch_size = 16
input_shape = (128, 512)  # you can adjust as needed

def get_inputs():
    x = 2.0 * torch.rand(batch_size, *input_shape) - 1.0
    return [x]

def get_init_inputs():
    return []
