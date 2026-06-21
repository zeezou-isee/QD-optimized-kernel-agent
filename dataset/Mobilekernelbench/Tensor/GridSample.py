import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    GridSample: sample input using a normalized grid.

    Semantics (PyTorch / ONNX-like):
        y = grid_sample(input, grid, mode, padding_mode, align_corners)

    Here:
        input: (N, C, H, W)
        grid:  (N, H_out, W_out, 2), values in [-1, 1]
    """

    def __init__(self, mode: str = "bilinear",
                       padding_mode: str = "zeros",
                       align_corners: bool = False):
        super(Model, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


# ======== Example input configuration ========

N, C, H, W = 1, 3, 32, 64
H_out, W_out = 4, 4

def get_inputs():
    x = torch.rand(N, C, H, W)
    # grid in [-1, 1]
    grid = 2.0 * torch.rand(N, H_out, W_out, 2) - 1.0
    return [x, grid]

def get_init_inputs():
    # Default bilinear / zeros / align_corners=False
    return ["bilinear", "zeros", False]
