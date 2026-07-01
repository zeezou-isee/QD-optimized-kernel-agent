import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start, limit, delta):
        return torch.arange(start, limit, delta)

start = torch.tensor(0.0)
limit = torch.tensor(1024.0)
delta = torch.tensor(1.0)

# ======== Example input configuration ========
def get_inputs() -> list:
    return [start, limit, delta]

def get_init_inputs():
    return []