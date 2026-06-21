import torch
import torch.nn as nn

class Model(nn.Module):
    """
    ReduceLogSumExp with keeping dimensions
    """
    def __init__(self, axes: int, keepdims: int = 1):
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = True
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        LogSumExp over specified dimension while keeping it as size 1
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor with specified dimension kept as 1 after LogSumExp
        """
        return torch.logsumexp(data, dim=self.axes, keepdim=self.keepdims)

axes = 1

def get_inputs():
    data = torch.randn(2, 3, 4)
    return [data]

def get_init_inputs():
    return [axes]