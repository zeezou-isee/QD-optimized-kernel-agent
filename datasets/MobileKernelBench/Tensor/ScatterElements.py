# ScatterElements.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a ScatterElements-like operation along a given axis.

    Semantics (assign mode):
        out = data.clone()
        out.scatter_(dim=axis, index=indices, src=updates)

    Requirements:
        - data, indices, updates have the same shape
        - indices contains valid indices along `axis`
    """

    def __init__(self, axis: int):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, data: torch.Tensor,
                      indices: torch.Tensor,
                      updates: torch.Tensor) -> torch.Tensor:
        assert data.shape == indices.shape == updates.shape, \
            "data, indices and updates must have the same shape"

        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)

        axis = self.axis if self.axis >= 0 else self.axis + data.dim()

        out = data.clone()
        out.scatter_(dim=axis, index=indices, src=updates)
        return out


# ======== Example input configuration ========

B, C, L = 8, 64, 64
data_shape = (B, C, L)
axis = 1

def get_inputs():
    data = torch.zeros(*data_shape)
    indices = torch.randint(low=0, high=C, size=data_shape, dtype=torch.int64)
    updates = torch.rand(*data_shape)

    return [data, indices, updates]

def get_init_inputs():
    return [axis]
