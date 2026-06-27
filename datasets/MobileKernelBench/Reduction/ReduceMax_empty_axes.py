import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMax over all axes (returns scalar)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maximum of all elements in the tensor
        
        Args:
            x: Input tensor
            
        Returns:
            Scalar tensor with maximum of all elements
        """
        return torch.max(x).unsqueeze(0) 
batch_size = 16
input_shape = (128, 128, 64)

def get_inputs() -> list:
    input = torch.rand(batch_size, *input_shape)
    return [input]

def get_init_inputs():
    return []