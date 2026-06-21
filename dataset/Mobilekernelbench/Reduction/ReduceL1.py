#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReduceL1 Operator - PyTorch Implementation

ONNX Opset Version: 18
Domain: ai.onnx
Category: reduction
Implementation Strategy: api_with_dim_handling

Semantics:
    Computes the L1 norm (sum of absolute values) of the input tensor's 
    elements along the provided axes. The resulting tensor has the same 
    rank as the input if keepdims equals 1. If keepdims equals 0, then 
    the resulting tensor has the reduced dimension pruned.
    
    Formula: output = sum(|input|, axes)
    
Auto-generated time: 2026-01-24
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple


class Model(nn.Module):
    """
    ReduceL1 Operator Implementation
    
    Computes the L1 norm of the input tensor's elements along the provided axes.
    L1 norm is defined as the sum of absolute values.
    
    Inputs:
        data: T (Single) - Input tensor with arbitrary shape
        axes: tensor(int64) (Optional) - List of integers indicating axes to reduce.
                                          If not provided, behavior depends on 
                                          noop_with_empty_axes attribute.
    
    Outputs:
        reduced: T (Single) - Reduced output tensor
    
    Attributes:
        noop_with_empty_axes: int (optional, default=0)
            - If 0: Empty axes means reduce over all dimensions
            - If 1: Empty axes means no operation (return input as-is)
        keepdims: int (optional, default=1)
            - If 1: Keep reduced dimensions with size 1
            - If 0: Remove reduced dimensions
    
    Mathematical Definition:
        For input tensor X and reduction axes A:
        L1(X, A) = Σ |X[i]| for i in A
    
    Examples:
        >>> # Example 1: Reduce all dimensions
        >>> model = Model(keepdims=1)
        >>> x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        >>> output = model(x)  # |−1|+|2|+|−3|+|4| = 10.0
        >>> print(output)  # tensor([[10.]])
        
        >>> # Example 2: Reduce specific axis
        >>> model = Model(keepdims=1)
        >>> x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        >>> axes = torch.tensor([0])
        >>> output = model(x, axes)  # [[|−1|+|−3|, |2|+|4|]] = [[4., 6.]]
        >>> print(output)
        
        >>> # Example 3: Reduce with keepdims=0
        >>> model = Model(keepdims=0)
        >>> x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        >>> axes = torch.tensor([1])
        >>> output = model(x, axes)  # [|−1|+|2|, |−3|+|4|] = [3., 7.]
        >>> print(output)
    """
    
    def __init__(self, noop_with_empty_axes: int = 0, keepdims: int = 1):
        """
        Initialize ReduceL1 operator
        
        Args:
            noop_with_empty_axes: Behavior when axes is empty
                                  0 = reduce all dimensions
                                  1 = no operation (return input)
            keepdims: Whether to keep reduced dimensions (1) or remove them (0)
        """
        super(Model, self).__init__()
        self.noop_with_empty_axes = noop_with_empty_axes
        self.keepdims = bool(keepdims)
    
    def forward(self, data: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of ReduceL1
        
        Computes L1 norm (sum of absolute values) along specified axes.
        
        Args:
            data: Input tensor to reduce
            axes: Optional tensor containing axes to reduce along.
                  If None, behavior depends on noop_with_empty_axes:
                  - If noop_with_empty_axes=0: reduce all axes
                  - If noop_with_empty_axes=1: return input unchanged
        
        Returns:
            Output tensor after L1 norm reduction
        
        Implementation Notes:
            L1 norm is computed as: sum(abs(data), axes)
            This is equivalent to: torch.sum(torch.abs(data), dim=axes, keepdim=keepdims)
            Or using torch.norm with p=1: torch.norm(data, p=1, dim=axes, keepdim=keepdims)
        """
        # Handle axes parameter
        if axes is None:
            # No axes provided
            if self.noop_with_empty_axes == 1:
                # No operation: return input unchanged
                return data
            else:
                # Reduce over all dimensions
                # Method 1: Using torch.norm with p=1
                l1_result = torch.norm(data, p=1)
                
                # Handle keepdims for scalar result
                if self.keepdims:
                    # Reshape to (1, 1, ..., 1) matching input rank
                    output_shape = [1] * len(data.shape)
                    l1_result = l1_result.reshape(output_shape)
                
                return l1_result
        else:
            # Convert axes tensor to tuple/list for torch operations
            if isinstance(axes, torch.Tensor):
                axes_list = axes.tolist()
                if isinstance(axes_list, int):
                    axes_list = [axes_list]
            else:
                axes_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]
            
            # Handle negative axes
            axes_list = [ax if ax >= 0 else len(data.shape) + ax for ax in axes_list]
            
            # Compute L1 norm along specified axes
            # Method 1: Using torch.norm with p=1 (recommended)
            l1_result = torch.norm(data, p=1, dim=tuple(axes_list), keepdim=self.keepdims)
            
            # Alternative Method 2: Using sum(abs(data))
            # l1_result = torch.sum(torch.abs(data), dim=tuple(axes_list), keepdim=self.keepdims)
            
            return l1_result


def get_inputs():
    """
    Generate test inputs
    
    Creates valid input tensors for ReduceL1 operator.
    Uses random values including negative numbers to test absolute value computation.
    
    Returns:
        List containing:
        - data: Input tensor with shape [2, 3, 4]
        - axes: (optional) Reduction axes
    """
    torch.manual_seed(42)
    # Generate data with both positive and negative values
    data = torch.randn(2, 3, 4)
    
    # Optional: specify axes to reduce
    # Uncomment to test with specific axes
    # axes = torch.tensor([1])  # Reduce along axis 1
    # return [data, axes]
    
    return [data]


def get_init_inputs():
    """
    Get initialization parameters for Model
    
    Returns:
        List of initialization arguments:
        - noop_with_empty_axes: 0 (reduce all when axes empty)
        - keepdims: 1 (keep reduced dimensions)
    """
    return [0, 1]

if __name__ == "__main__":
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    
    print("=" * 60)
    print(f"ReduceL1 operator test")
    print("=" * 60)
    
    try:
        with torch.no_grad():
            outputs = model(*inputs)
        
        print(f"\nInput count: {len(inputs)}")
        for i, inp in enumerate(inputs):
            print(f"  Input {i}: shape={inp.shape}, dtype={inp.dtype}")
        
        if isinstance(outputs, (list, tuple)):
            print(f"\nOutput count: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        else:
            print(f"\nOutput: shape={outputs.shape}, dtype={outputs.dtype}")
        
        print("\n✅ Test passed!")
    except NotImplementedError as e:
        print(f"\n⚠️  Operator is not implemented yet: {e}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")