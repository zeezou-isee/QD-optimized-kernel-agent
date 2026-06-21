"""Validate ncnn base (non-optimized) layer implementations against the
MobileKernelBench PyTorch dataset, using the LayerOracle runner.

For each dataset model that maps cleanly to a single ncnn layer, we:
  1. run the PyTorch model on its own get_inputs() -> reference output,
  2. feed the SAME single sample (batch dropped) through the ncnn base kernel
     via LayerOracle (create via direct cpp compile, opt all-off = naive path),
  3. allclose-verify.

Operators without a clean 1:1 mapping are skipped with a reason.
Writes layer_oracle/VALIDATION_REPORT.md.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np

AGENT = Path(__file__).resolve().parent.parent   # .../opgen
sys.path.insert(0, str(AGENT))

from config import KERNELGEN_ROOT
from layer_oracle import LayerOracle, torch_to_ncnn_input

DATASET = KERNELGEN_ROOT / "MobileKernelBench_git" / "dataset" / "Mobilekernelbench"
NCNN = KERNELGEN_ROOT / "ncnn"
LAYER = NCNN / "src" / "layer"


def U(op_type: int) -> dict:
    return dict(layer="UnaryOp", cpp="unaryop.cpp", header="unaryop.h",
                cls="UnaryOp", params={0: op_type})


def B(op_type: int) -> dict:
    return dict(layer="BinaryOp", cpp="binaryop.cpp", header="binaryop.h",
                cls="BinaryOp", params={0: op_type})


def simple(cls: str, cpp: str, params=None) -> dict:
    return dict(layer=cls, cpp=cpp, header=cpp.replace(".cpp", ".h"), cls=cls, params=params or {})


# dataset file (relative) -> ncnn mapping. params_from_init / weights_from_model
# are filled by special handlers below.
MAPPING: dict[str, dict] = {
    # ---- Unary (UnaryOp op_type) ----
    "Unary/Abs.py": U(0), "Unary/Neg.py": U(1), "Unary/Floor.py": U(2),
    "Unary/Ceil.py": U(3), "Unary/Sqrt.py": U(5), "Unary/Exp.py": U(7),
    "Unary/Log.py": U(8), "Unary/Reciprocal.py": U(15), "Unary/Round.py": U(18),
    "Unary/Sign.py": U(20), "Unary/Erf.py": simple("Erf", "erf.cpp"),
    # ---- Trigonometry (UnaryOp) ----
    "Trigonometry/Sin.py": U(9), "Trigonometry/Cos.py": U(10), "Trigonometry/Tan.py": U(11),
    "Trigonometry/Asin.py": U(12), "Trigonometry/Acos.py": U(13), "Trigonometry/ATan.py": U(14),
    "Trigonometry/Tanh.py": U(16), "Trigonometry/Sinh.py": U(22), "Trigonometry/Asinh.py": U(23),
    "Trigonometry/Cosh.py": U(24), "Trigonometry/Acosh.py": U(25), "Trigonometry/Atanh.py": U(26),
    # ---- Activation ----
    "Activation/Relu.py": simple("ReLU", "relu.cpp", {0: 0.0}),
    "Activation/HardSigmoid.py": simple("HardSigmoid", "hardsigmoid.cpp", {0: 1.0 / 6, 1: 0.5}),
    "Activation/ELU.py": dict(layer="ELU", cpp="elu.cpp", header="elu.h", cls="ELU", params_from_init=lambda init: {0: float(init[0])}),
    "Activation/Celu.py": dict(layer="CELU", cpp="celu.cpp", header="celu.h", cls="CELU", params_from_init=lambda init: {0: float(init[0])}),
    "Activation/Softplus_2d.py": simple("Softplus", "softplus.cpp"),
    "Activation/Softplus_3d.py": simple("Softplus", "softplus.cpp"),
    "Activation/PRelu.py": dict(layer="PReLU", cpp="prelu.cpp", header="prelu.h", cls="PReLU", prelu=True),
    # ---- Binary (BinaryOp, two inputs) ----
    "Binary/Add.py": B(0), "Binary/Sub.py": B(1), "Binary/Mul.py": B(2),
    "Binary/Div.py": B(3), "Binary/Min.py": B(5), "Binary/Pow.py": B(6),
}

# explicitly skipped (no clean 1:1 base-layer mapping for this harness)
SKIP_REASON = {
    "Activation/Softmax.py": "axis/层布局语义需专门映射",
    "Activation/LogSoftmax.py": "ncnn 无直接 LogSoftmax 基础层",
    "Activation/LogSoftmax_axis_0.py": "同上",
    "Activation/LogSoftmax_axis_1.py": "同上",
    "Activation/LogSoftmax_axis_2.py": "同上",
    "Binary/Mod.py": "torch fmod/remainder 与 ncnn FMOD 语义需核对",
    "Binary/Mul_bcast.py": "广播:两输入 rank 不同,去 batch 后 ncnn 广播布局与 torch 不一致,需专门处理",
}


def load_model(py_path: Path):
    spec = importlib.util.spec_from_file_location("ds_model", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = mod.Model(*init) if init else mod.Model()
    model.eval()
    return mod, model, init


def run_one(oc: LayerOracle, rel: str, spec: dict) -> dict:
    import torch
    py = DATASET / rel
    mod, model, init = load_model(py)
    inputs = mod.get_inputs()
    with torch.no_grad():
        ref = model(*inputs)
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    ref_np = ref.detach().numpy()

    # single-sample compare (ncnn has no batch dim)
    ncnn_inputs = [torch_to_ncnn_input(t.detach().numpy()) for t in inputs]
    reference = ref_np[0] if ref_np.ndim >= 2 else ref_np

    params = dict(spec.get("params") or {})
    weights = []
    if spec.get("params_from_init"):
        params = spec["params_from_init"](init)
    if spec.get("prelu"):
        slope = list(model.state_dict().values())[0].detach().numpy().reshape(-1)
        params = {0: int(slope.size)}
        weights = [slope]

    verdict = oc.verify(
        candidate_cpp=LAYER / spec["cpp"],
        class_name=spec["cls"], header=spec["header"],
        params=params, inputs=ncnn_inputs, weights=weights,
        reference=reference, tol=2e-3,
    )
    return {
        "op": rel, "layer": spec["layer"],
        "in_shape": list(ncnn_inputs[0].shape), "ref_shape": list(reference.shape),
        "passed": verdict.passed, "max_diff": verdict.max_diff,
        "error": verdict.error, "detail": verdict.detail,
    }


def main() -> None:
    oc = LayerOracle(ncnn_root=NCNN)
    results = []
    for rel, spec in MAPPING.items():
        print(f"[validate] {rel} -> {spec['layer']}", flush=True)
        try:
            results.append(run_one(oc, rel, spec))
        except Exception as exc:  # noqa: BLE001
            results.append({"op": rel, "layer": spec.get("layer", "?"), "passed": False,
                            "error": f"{type(exc).__name__}: {exc}", "detail": "",
                            "in_shape": None, "ref_shape": None, "max_diff": None})
            traceback.print_exc()

    write_report(results)


def write_report(results: list[dict]) -> None:
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    lines = []
    lines.append("# ncnn 基础层实现 验证报告(vs MobileKernelBench PyTorch 数据集)")
    lines.append("")
    lines.append("用 `LayerOracle`(方案A:直接编译 `ncnn/src/layer/*.cpp` 基础实现 + 链接 libncnn.a,"
                 "关闭 packing/fp16 等优化=naive 路径)对比 PyTorch 真值。单样本对比(去 batch 维),tol=2e-3。")
    lines.append("")
    lines.append(f"- 数据来源:`MobileKernelBench_git/dataset/Mobilekernelbench`")
    lines.append(f"- 被测:`ncnn/src/layer/*.cpp` 基础(非优化)实现")
    lines.append(f"- 结果:**{len(passed)}/{len(results)} 通过**;跳过 {len(SKIP_REASON)} 个(无干净 1:1 映射)")
    lines.append("")
    lines.append("## 通过 ✅")
    lines.append("| 算子 | ncnn layer | 输入shape | max_diff |")
    lines.append("|---|---|---|---|")
    for r in passed:
        lines.append(f"| {r['op']} | {r['layer']} | {r['in_shape']} | {r['max_diff']:.2e} |")
    if failed:
        lines.append("")
        lines.append("## 未通过 / 出错 ❌")
        lines.append("| 算子 | ncnn layer | 现象 |")
        lines.append("|---|---|---|")
        for r in failed:
            msg = r["error"] or r["detail"] or "allclose 失败"
            lines.append(f"| {r['op']} | {r['layer']} | {msg[:120].replace(chr(10),' ')} |")
    lines.append("")
    lines.append("## 跳过(无干净 1:1 基础层映射)")
    lines.append("| 算子 | 原因 |")
    lines.append("|---|---|")
    for k, v in SKIP_REASON.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("> 说明:跳过项不代表 ncnn 不支持,而是需要专门的 param/权重/轴 映射(如 Softmax 轴、"
                 "Conv/Norm 的权重布局、Reduction 的 keepdim 等),不在本次「基础逐元素/激活」验证范围内。")
    out = Path(__file__).resolve().parent / "VALIDATION_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written: {out}")
    print(f"PASSED {len(passed)}/{len(results)}")


if __name__ == "__main__":
    main()
