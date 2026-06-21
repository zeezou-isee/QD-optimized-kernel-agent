import torch
import torch.nn as nn

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

class Model(nn.Module):
    """
    Cast: cast the input tensor to a target dtype.

    Semantics (ONNX Cast-like):
        y = Cast(x, to=target_dtype)

    Parameters:
        to_dtype (str): one of {"float32","float16","float64","int32","int64","bool"}
    """

    def __init__(self, to_dtype: str = "int32"):
        super(Model, self).__init__()
        assert to_dtype in _DTYPE_MAP, f"Unsupported dtype: {to_dtype}"
        self.to_dtype_name = to_dtype
        self.to_dtype = _DTYPE_MAP[to_dtype]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.to_dtype)


# ======== Example input configuration ========

batch_size = 8
input_shape = (512, 256)

def get_inputs():
    x = torch.randn(batch_size, *input_shape, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return ["float16"]
