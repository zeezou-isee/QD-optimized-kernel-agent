import torch
import torch.nn as nn
from typing import Tuple, Optional

class Model(nn.Module):
    """
    ReduceL1 over multiple axes
    
    Reduces along multiple dimensions simultaneously.
    
    Example:
        >>> model = Model(axes=(0, 2), keepdims=0)
        >>> x = torch.randn(2, 3, 4)
        >>> axes_tensor = torch.tensor([0, 2])
        >>> output = model(x, axes_tensor)
        >>> print(output.shape)  # torch.Size([3])
    """
    
    def __init__(self, axes: Tuple[int, ...], keepdims: int = 0):
        super(Model, self).__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
    
    def forward(self, data: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute L1 norm over multiple specified dimensions
        
        Args:
            data: Input tensor
            axes: Axes to reduce (optional)
            
        Returns:
            Tensor with L1 norm over multiple dimensions
        """
        if axes is None:
            axes_list = self.axes
        else:
            axes_list = tuple(axes.tolist())
        
        return torch.norm(data, p=1, dim=axes_list, keepdim=self.keepdims)

axes = (0, 2)
keepdims = 0

def get_inputs():
    torch.manual_seed(42)
    data = torch.randn(2, 3, 4)
    return [data]

def get_init_inputs():
    return [axes, keepdims]