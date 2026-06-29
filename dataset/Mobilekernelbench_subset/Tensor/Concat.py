import torch
import torch.nn as nn
from typing import List, Tuple


class Model(nn.Module):
    """
    Concat Operator Implementation
    
    Concatenates a list of tensors into a single tensor along a specified axis.
    
    Inputs:
        inputs: T (Variadic) - A list of tensors to concatenate.
                               All tensors must have the same shape except
                               along the concatenation axis.
    
    Outputs:
        concat_result: T (Single) - The concatenated tensor
    
    Attributes:
        axis: int (required, default=0) - The axis along which to concatenate.
                                          Negative values are supported.
    
    Examples:
        >>> # Example 1: Concatenate along axis 0
        >>> model = Model(axis=0)
        >>> x1 = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
        >>> x2 = torch.tensor([[5, 6]])           # shape: (1, 2)
        >>> output = model(x1, x2)                # shape: (3, 2)
        >>> print(output)
        tensor([[1, 2],
                [3, 4],
                [5, 6]])
        
        >>> # Example 2: Concatenate along axis 1
        >>> model = Model(axis=1)
        >>> x1 = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
        >>> x2 = torch.tensor([[5], [6]])         # shape: (2, 1)
        >>> output = model(x1, x2)                # shape: (2, 3)
        >>> print(output)
        tensor([[1, 2, 5],
                [3, 4, 6]])
        
        >>> # Example 3: Negative axis
        >>> model = Model(axis=-1)  # Same as axis=1 for 2D tensors
        >>> x1 = torch.tensor([[1], [2]])
        >>> x2 = torch.tensor([[3], [4]])
        >>> output = model(x1, x2)
        >>> print(output)
        tensor([[1, 3],
                [2, 4]])
    """
    
    def __init__(self, axis: int = 0):
        """
        Initialize Concat operator
        
        Args:
            axis: Axis along which to concatenate tensors.
                  Can be negative (counts from the end).
        """
        super(Model, self).__init__()
        self.axis = axis
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Concat operator
        
        Args:
            *inputs: Variable number of input tensors to concatenate.
                     All tensors must have the same shape except along
                     the concatenation axis.
        
        Returns:
            Concatenated tensor
        
        Raises:
            RuntimeError: If inputs have incompatible shapes
            ValueError: If no inputs are provided
        
        Note:
            - At least one input tensor is required
            - All input tensors must have the same number of dimensions
            - All dimensions must match except the concatenation axis
        """
        if len(inputs) == 0:
            raise ValueError("Concat requires at least one input tensor")
        
        # Convert inputs tuple to list
        input_list = list(inputs)
        
        # Validate inputs
        if len(input_list) == 1:
            # Single input - return as-is
            return input_list[0]
        
        # Concatenate along specified axis
        return torch.cat(input_list, dim=self.axis)

batch_size = 16
dim1 = 128
dim2 = 64

# ======== Example input configuration ========
def get_inputs():
    """
    Generate test inputs
    
    Creates multiple tensors to concatenate.
    Returns a list of tensors with compatible shapes.
    
    Returns:
        List of input tensors to concatenate
    """
    torch.manual_seed(42)
    
    # Create multiple tensors with compatible shapes
    # Shape: (batch_size, dim1, dim2) - will concatenate along axis 0
    input1 = torch.randn(batch_size, dim1, dim2)
    input2 = torch.randn(batch_size, dim1, dim2)
    input3 = torch.randn(batch_size, dim1, dim2)  # Different size along axis 0
    
    return [input1, input2, input3]


def get_init_inputs():
    """
    Get initialization parameters for Model
    
    Returns:
        List containing:
        - axis: Concatenation axis (default=0)
    """
    return [0]  # Concatenate along axis 0
