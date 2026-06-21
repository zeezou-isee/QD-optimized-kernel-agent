import torch
import torch.nn as nn
from typing import List, Tuple


class Model(nn.Module):
    """
    A model that applies Layer Normalization for vision transformers.
    
    Number of inputs: 1
    Implementation type: direct
    
    Semantics:
        Normalizes patches in vision transformer style.
        Normalizes over (num_patches, embed_dim).
    """
    
    def __init__(self, normalized_shape: Tuple[int, int], eps: float = 1e-6, 
                 elementwise_affine: bool = True):
        """
        Args:
            normalized_shape: Tuple of (num_patches, embed_dim) to normalize
            eps: Small value added to variance for numerical stability
            elementwise_affine: If True, learnable affine parameters
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor with shape [batch_size, num_patches, embed_dim]
        
        Returns:
            Output tensor with same shape as input
        """
        return self.ln(x)


# ======== Example input configuration ========

batch_size = 1
num_patches = 196  # 14x14 patches for 224x224 image with 16x16 patch size
embed_dim = 768
eps = 1e-5
elementwise_affine = True

def get_inputs():
    x = torch.randn(batch_size, num_patches, embed_dim)
    return [x]

def get_init_inputs():
    return [embed_dim, eps, elementwise_affine]
