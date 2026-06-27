import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceMean over spatial dimensions (Global Average Pooling)
    """
    def __init__(self, dims: tuple = (2, 3), keepdim: bool = False):
        super(Model, self).__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean over spatial dimensions (height and width)
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tensor with spatial dimensions reduced
        """
        return torch.mean(x, dim=self.dims, keepdim=self.keepdim)

batch_size = 8
input_shape = (128, 64, 64)
dims = (2, 3)
keepdim = False

def get_inputs() -> list:
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dims, keepdim]