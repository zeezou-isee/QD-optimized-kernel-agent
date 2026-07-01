# Gather.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a Gather operation along a given axis.

    Semantics:
        y = gather(params, axis, indices)

    Here we implement the common case where `indices` is 1D:
        params: Tensor of shape (d0, d1, ..., d_{k-1})
        axis:   integer in [0, k-1]
        indices: 1D Long tensor of shape (n,)

    Output shape:
        (d0, ..., d_{axis-1}, n, d_{axis+1}, ..., d_{k-1})
    """

    def __init__(self, axis: int):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # indices must be 1D Long tensor
        assert indices.dim() == 1, "This standard case expects 1D indices"
        assert indices.dtype in (torch.int32, torch.int64)
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)

        axis = self.axis if self.axis >= 0 else self.axis + params.dim()
        return torch.index_select(params, dim=axis, index=indices)


# ======== Example input configuration ========

# params shape: (B, C, D)
B, C, D = 8, 64, 64
axis = 1  # gather along C-dimension

def get_inputs():
    params = torch.rand(B, C, D)
    # indices in [0, C)
    idx = torch.randint(low=0, high=C, size=(3,), dtype=torch.int64)
    return [params, idx]

def get_init_inputs():
    # Constructor arguments for Model(...)
    return [axis]
