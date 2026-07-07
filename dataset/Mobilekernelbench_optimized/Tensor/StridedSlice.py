# Slice.py
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ONNX Slice-equivalent model.
    Extracts a single slice from the input tensor.
    """

    def __init__(self, axis: int, start: int, end: int):
        super().__init__()
        self.axis = axis
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor):
        axis = self.axis if self.axis >= 0 else self.axis + x.dim()

        # Build slicing objects for all dimensions
        slices = [slice(None)] * x.dim()
        slices[axis] = slice(self.start, self.end)

        return x[tuple(slices)]

# ======== Example input configuration ========

batch_size = 32
input_shape = (8, 64, 64)   # C, H, W

# Slice configuration (ONNX Slice semantics)
axis = 1
start = 3
end = 6   # end is exclusive, matches ONNX Slice

def get_inputs():
    # Input shape: (N, C, H, W)
    x = torch.rand(batch_size, *input_shape)
    return [x]

def get_init_inputs():
    # Model __init__(axis, start, end)
    return [axis, start, end]

