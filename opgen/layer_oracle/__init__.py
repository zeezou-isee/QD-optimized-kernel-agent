"""Reusable ncnn layer oracle: run a candidate kernel .cpp and verify vs PyTorch."""

from .oracle import (
    LayerOracle,
    OracleResult,
    read_bin,
    write_bin,
    torch_to_ncnn_input,
    parse_pnnx_input_squeeze,
    pnnx_driven_ncnn_inputs,
)
from .net_oracle import (
    NetOracle,
    InstallHandle,
    parse_ncnn_io,
    retarget_param_layer,
    retarget_param_output_layer,
    retarget_param_output_file,
    retarget_param_file,
)
from .vulkan_oracle import VulkanLayerOracle
from .device_oracle import DeviceOracle, VulkanDeviceOracle
from .failure_taxonomy import classify_failure

__all__ = [
    "LayerOracle",
    "OracleResult",
    "classify_failure",
    "read_bin",
    "write_bin",
    "torch_to_ncnn_input",
    "parse_pnnx_input_squeeze",
    "pnnx_driven_ncnn_inputs",
    "NetOracle",
    "InstallHandle",
    "parse_ncnn_io",
    "retarget_param_layer",
    "retarget_param_output_layer",
    "retarget_param_output_file",
    "retarget_param_file",
    "VulkanLayerOracle",
    "DeviceOracle",
    "VulkanDeviceOracle",
]
