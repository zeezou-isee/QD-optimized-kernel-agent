import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes element-wise minimum of two tensors.

    Semantics:
        y = min(x1, x2)
        Computes the element-wise minimum of two input tensors.
        Input shape: Any broadcastable shapes
        Output shape: Broadcasted shape of inputs
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: First input tensor
            x2: Second input tensor
        
        Returns:
            Tensor containing element-wise minimum
        """
        return torch.min(x1, x2)


# ======== Example input configuration ========

batch_size = 16
dim1 = 64
dim2 = 512

def get_inputs():
    x1 = torch.randn(batch_size, dim1, dim2)
    x2 = torch.randn(batch_size, dim1, dim2)
    return [x1, x2]

def get_init_inputs():
    return []

