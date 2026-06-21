import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a Tile operation on the input tensor.

    Given an input tensor x and a list of multiples, this model
    repeats x along each dimension according to `multiples`,
    equivalent to ONNX::Tile and torch.Tensor.repeat.

    Input:
        x: Tensor of shape (batch_size, *input_shape)

    Parameters:
        multiples (tuple or list of int):
            The repetition factor for each dimension. Its length must
            equal x.dim() (i.e. number of dimensions of the input).
    """

    def __init__(self, multiples):
        super(Model, self).__init__()
        self.multiples = tuple(int(m) for m in multiples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == len(self.multiples), \
            f"Expected multiples length {len(self.multiples)} to match x.dim()={x.dim()}"
        return x.repeat(*self.multiples)


# ======== Example input configuration ========

batch_size = 16
input_shape = (64, 64)   # C, W

multiples = (1, 2, 3)

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [multiples]
