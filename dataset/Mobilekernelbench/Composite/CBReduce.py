import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x):
        y = self.conv(x)
        return torch.sum(y, dim=1, keepdim=False)
batch_size = 8
def get_inputs():
    return [torch.rand(batch_size, 3, 32, 32)]
def get_init_inputs():
    return []
