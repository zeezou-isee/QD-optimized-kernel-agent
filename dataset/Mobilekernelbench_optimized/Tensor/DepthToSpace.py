import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a depth-to-space (pixel shuffle) operation.

    This is equivalent to the DepthToSpace operator with a given block size
    in NCHW layout. Input shape is (N, C * block_size^2, H, W) and the output
    shape is (N, C, H * block_size, W * block_size).

    Parameters:
        block_size (int): The spatial upscaling factor (DepthToSpace blockSize).
    """

    def __init__(self, block_size: int):
        super(Model, self).__init__()
        self.block_size = block_size
        # In ONNX export, this will map to a DepthToSpace op (mode = DCR).
        self.op = nn.PixelShuffle(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ======== Example input configuration ========

# DepthToSpace: (N, C_in, H_in, W_in)
# C_in = C_out * block_size^2
batch_size = 4
block_size = 4

out_channels = 8       # C_out
out_height = 64        # H_out
out_width = 64         # W_out

in_channels = out_channels * (block_size ** 2)  # C_in = C_out * r^2
in_height = out_height // block_size            # H_in = H_out / r
in_width = out_width // block_size              # W_in = W_out / r

input_shape = (in_channels, in_height, in_width)


def get_inputs():
    # NCHW: (batch_size, C_in, H_in, W_in)
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    # Constructor args for Model(...)
    return [block_size]
