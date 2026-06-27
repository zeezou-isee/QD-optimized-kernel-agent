import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceSum over all axes (returns scalar)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum all elements in the tensor
        
        Args:
            x: Input tensor
            
        Returns:
            Scalar tensor with sum of all elements
        """
        x = torch.sum(x, dim=(2, 3))   
        x = torch.sum(x, dim=1)        
        return x

batch_size = 1
input_shape = (64, 32, 8)

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []