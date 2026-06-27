import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Batch Normalization with custom momentum.
    """
    def __init__(self, num_features: int, momentum: float = 0.1):
        """
        Initializes the BatchNorm layer with custom momentum.

        Args:
            num_features (int): Number of features in the input tensor.
            momentum (float, optional): Momentum for running mean and variance computation.
                                        Formula: running_stat = (1 - momentum) * running_stat + momentum * batch_stat
                                        Defaults to 0.1.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization with custom momentum to the input tensor.

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
momentum = 0.2  # Custom momentum value (higher than default 0.1)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, momentum]