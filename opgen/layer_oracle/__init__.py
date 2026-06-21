"""Reusable ncnn layer oracle: run a candidate kernel .cpp and verify vs PyTorch."""

from .oracle import (
    LayerOracle,
    OracleResult,
    read_bin,
    write_bin,
    torch_to_ncnn_input,
)
from .net_oracle import NetOracle, InstallHandle, parse_ncnn_io

__all__ = [
    "LayerOracle",
    "OracleResult",
    "read_bin",
    "write_bin",
    "torch_to_ncnn_input",
    "NetOracle",
    "InstallHandle",
    "parse_ncnn_io",
]
