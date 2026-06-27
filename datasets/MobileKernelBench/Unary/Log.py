import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Any

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log(input)

batch_size = 8
dim1 = 64
dim2 = 256

# ======== Example input configuration ========
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2) + 0.1
    return [x]

def get_init_inputs():
    return []
