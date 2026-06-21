import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a reshape operation on the input tensor.

    The input is assumed to be of shape (batch_size, *input_shape),
    and it will be reshaped to `target_shape`.

    Parameters:
        target_shape (tuple or list of int): The target shape passed to
            torch.reshape. It must be compatible with the input number of
            elements. You can use -1 for one dimension to let PyTorch infer it.
    """

    def __init__(self, target_shape):
        super(Model, self).__init__()
        self.target_shape = tuple(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, self.target_shape)


# ======== Example input configuration ========

batch_size = 16
input_shape = (8, 64, 64)  # C, H, W

target_shape = (batch_size, -1)

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [target_shape]
