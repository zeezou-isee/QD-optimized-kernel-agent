import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the determinant (Det) of the input tensor.

    Semantics:
        y = det(x)
        Computes the determinant of each square matrix in the input tensor.
        Input shape: (..., n, n) where the last two dimensions form square matrices.
        Output shape: (...) with determinant values for each matrix.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., n, n) containing square matrices.
        
        Returns:
            Tensor of shape (...) containing determinant of each matrix.
        """
        return torch.linalg.det(x)


# ======== Example input configuration ========

matrix_size = 512  # n x n square matrix
input_shape = (matrix_size, matrix_size)

def get_inputs():
    # Generate random square matrices
    x = 2.0 * torch.rand(*input_shape) - 1.0
    return [x]

def get_init_inputs():
    return []