import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, depth, on_value=1, off_value=0, axis=-1):
        super(Model, self).__init__()
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        one_hot_result = F.one_hot(indices, num_classes=self.depth)
        one_hot_result = one_hot_result * (self.on_value - self.off_value) + self.off_value

        if self.axis != -1:
            one_hot_result = one_hot_result.permute(*range(one_hot_result.dim() - 1), self.axis, -1)
        
        return one_hot_result

batch_size = 16
dim1 = 128
dim2 = 1024
depth = 10

# ======== Example input configuration ========
def get_inputs() -> list:
    x = torch.randint(0, depth, (batch_size, dim1))
    return [x]

def get_init_inputs():
    return [depth]
