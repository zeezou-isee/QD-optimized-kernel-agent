import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):
    """
    Simple model that performs ROI Align.
    """
    def __init__(self, output_size: tuple, spatial_scale: float, sampling_ratio: int, aligned: bool):
        super(Model, self).__init__()
        self.roi_align = torchvision.ops.RoIAlign(  
            output_size=output_size,  
            spatial_scale=spatial_scale,  
            sampling_ratio=sampling_ratio,  
            aligned=aligned  
        )  
        self.rois = nn.Parameter(  
            torch.tensor([[0, 2.5, 2.5, 12.5, 12.5]], dtype=torch.float), 
            requires_grad=False  
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.roi_align(x, self.rois) 

batch_size = 1
channels = 16
height = 128
width = 128

def get_inputs():  
    x = torch.rand(batch_size, channels, height, width)  
    return [x]  
  
def get_init_inputs():  
    return [(7, 7), 1.0, 2, True]