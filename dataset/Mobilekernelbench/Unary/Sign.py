import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Any

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)

batch_size = 8
dim1 = 128
dim2 = 256

# ======== Example input configuration ========
def get_inputs():
    input = torch.randn(batch_size, dim1, dim2)
    return [input]

def get_init_inputs():
    return []
