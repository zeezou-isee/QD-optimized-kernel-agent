import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Any

class Model(nn.Module):
    def __init__(self, axis: int = 0, sorted: int = 1):
        super(Model, self).__init__()
        self.axis = axis
        self.sorted = sorted
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        axis = self.axis 
        sorted_out = 1
        
        output, inverse_indices = torch.unique(x, dim=axis, return_inverse=True, sorted=bool(sorted_out))
        return output, inverse_indices

batch_size = 32
dim1 = 512
dim2 = 512

# ======== Example input configuration ========
def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [0, 1]
