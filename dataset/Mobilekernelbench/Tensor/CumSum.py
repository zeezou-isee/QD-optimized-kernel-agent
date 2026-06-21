import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that applies cumulative sum (CumSum) to the input tensor.

    Semantics:
        y = cumsum(x, dim=axis)
        Computes the cumulative sum of elements along the specified dimension.
    """

    def __init__(self, axis: int = -1):
        """
        Args:
            axis: The dimension along which to compute cumulative sum. Default is -1 (last dimension).
        """
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, dim=self.axis)


# ======== Example input configuration ========
batch_size = 16
input_shape = (32, 32)

def get_inputs():
    x = 2.0 * torch.rand(batch_size, *input_shape) - 1.0
    return [x]

def get_init_inputs():
    return []