import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMin over all axes (returns scalar)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Minimum of all elements in the tensor
        
        Args:
            x: Input tensor
            
        Returns:
            Scalar tensor with minimum of all elements
        """
        return torch.min(x).unsqueeze(0)

batch_size = 8
input_shape = (64, 32, 32)

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []