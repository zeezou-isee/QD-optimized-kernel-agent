import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Any

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)

batch_size = 8
dim1 = 256
dim2 = 512

# ======== Example input configuration ========
def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return []
