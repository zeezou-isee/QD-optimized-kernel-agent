# GatherElements.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a GatherElements operation along a given axis.

    Semantics (same as ONNX GatherElements / torch.gather):
        output = gather_elements(data, axis, indices)

    Requirements:
        - data and indices have the same shape
        - indices contains valid indices along `axis`
    """

    def __init__(self, axis: int):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert data.shape == indices.shape, "data and indices must have the same shape"
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)

        axis = self.axis if self.axis >= 0 else self.axis + data.dim()
        return torch.gather(data, dim=axis, index=indices)


# ======== Example input configuration ========

# data shape: (B, C, D)
B, C, D = 16, 64, 64
axis = 1  # gather along C-dimension

def get_inputs():
    data = torch.rand(B, C, D)
    # indices must be in [0, C) for axis=1
    idx = torch.randint(low=0, high=C, size=(B, C, D), dtype=torch.int64)
    return [data, idx]

def get_init_inputs():
    return [axis]
