"""CLI / self-test for the reusable LayerOracle.

Default action runs a built-in self-test on ncnn Convolution1D: build a PyTorch
reference (F.conv1d), run the same op through the ncnn kernel via the generic
runner, and allclose-verify (PyTorch is the oracle).

    python run_layer_oracle.py            # run the conv1d self-test (needs torch + libncnn.a)

Use LayerOracle directly in code for other layers:

    from layer_oracle import LayerOracle, torch_to_ncnn_input
    oc = LayerOracle()
    verdict = oc.verify(candidate_cpp=..., class_name=..., header=...,
                        params={...}, inputs=[x_ncnn], weights=[w, b], reference=ref_np)
"""

from __future__ import annotations

import sys
from pathlib import Path

# bootstrap opgen flat-import paths (we live in opgen/cli/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # need EndtoEnd... so `import opgen` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # opgen/ for flat imports
import opgen as _opgen; _opgen.bootstrap_paths()

from config import KERNELGEN_ROOT
from layer_oracle import LayerOracle, torch_to_ncnn_input


def selftest_conv1d() -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F

    ncnn_root = KERNELGEN_ROOT / "ncnn"
    candidate = ncnn_root / "src" / "layer" / "convolution1d.cpp"

    # config
    in_ch, out_ch, k, length = 2, 4, 3, 10
    stride, pad, dilation = 1, 0, 1
    torch.manual_seed(0)
    x = torch.randn(1, in_ch, length)
    weight = torch.randn(out_ch, in_ch, k)
    bias = torch.randn(out_ch)

    ref = F.conv1d(x, weight, bias, stride=stride, padding=pad, dilation=dilation)
    ref_np = ref.detach().numpy()  # (1, out_ch, out_len)

    oc = LayerOracle(ncnn_root=ncnn_root)
    verdict = oc.verify(
        candidate_cpp=candidate,
        class_name="Convolution1D",
        header="convolution1d.h",
        params={
            0: out_ch,      # num_output
            1: k,           # kernel_w
            2: dilation,    # dilation_w
            3: stride,      # stride_w
            4: pad,         # pad_left
            5: 1,           # bias_term
            6: out_ch * in_ch * k,  # weight_data_size
        },
        inputs=[torch_to_ncnn_input(x.numpy())],        # (in_ch, length)
        weights=[weight.detach().numpy().reshape(-1), bias.detach().numpy()],
        reference=ref_np,
        tol=1e-3,
    )

    print("=== Convolution1D oracle self-test ===")
    print("compile:", "ok" if verdict.error == "" else verdict.error)
    print("run rc:", verdict.return_code)
    if verdict.outputs:
        print("ncnn out shape:", verdict.outputs[0].shape, "| torch ref shape:", ref_np.shape)
    print("verdict:", "PASS ✅" if verdict.passed else "FAIL ❌", "|", verdict.detail)
    if not verdict.passed:
        print("--- run log tail ---")
        print("\n".join(verdict.run_log.splitlines()[-15:]))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(selftest_conv1d())
