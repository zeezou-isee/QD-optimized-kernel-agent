import torch
import torch.nn as nn
from typing import Optional

class Model(nn.Module):
    """
    ReduceL1 on 2D input
    
    Common for matrix operations.
    
    Example:
        >>> model = Model(axes=1, keepdims=0)
        >>> x = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
        >>> axes_tensor = torch.tensor([1])
        >>> output = model(x, axes_tensor)
        >>> print(output)  # tensor([6., 15.])  # Row-wise L1 norms
    """
    
    def __init__(self, axes: int, keepdims: int = 0):
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
    
    def forward(self, data: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute L1 norm over 2D tensor
        
        Args:
            data: 2D input tensor
            axes: Axes to reduce (optional)
            
        Returns:
            Tensor with specified dimension reduced
        """
        if axes is None:
            axes = self.axes
        else:
            axes = axes.item() if axes.numel() == 1 else axes.tolist()
        
        return torch.norm(data, p=1, dim=axes, keepdim=self.keepdims)

rows = 64
cols = 128
axes = 1
keepdims = 0

def get_inputs():
    torch.manual_seed(42)
    data = torch.randn(rows, cols)
    return [data]

def get_init_inputs():
    return [axes, keepdims]