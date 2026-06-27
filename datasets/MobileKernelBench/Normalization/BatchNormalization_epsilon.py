import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Batch Normalization with custom epsilon.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the BatchNorm layer with custom epsilon.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): Small value added to denominator for numerical stability. 
                                   Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization with custom epsilon to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 1
features = 32
dim1 = 128
dim2 = 128
epsilon = 1e-3  # Custom epsilon value (larger than default 1e-5)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, epsilon]