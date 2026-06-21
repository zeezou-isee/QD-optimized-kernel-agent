import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceLogSumExp on 2D input
    """
    def __init__(self, axes: int, keepdims: int = 0):
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = keepdims
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        LogSumExp over 2D tensor
        
        Args:
            data: 2D input tensor
            
        Returns:
            Tensor with specified dimension reduced
        """
        return torch.logsumexp(data, dim=self.axes, keepdim=bool(self.keepdims))

rows = 64
cols = 128
axes = 1
keepdims = 0

def get_inputs():
    data = torch.randn(rows, cols)
    return [data]

def get_init_inputs():
    return [axes, keepdims]