import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(x)

batch_size = 2
dim1 = 64
dim2 = 128

# ======== Example input configuration ========
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2) + 10.0
    return [x]

def get_init_inputs():
    return []
