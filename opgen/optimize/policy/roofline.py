"""Roofline diagnosis + regime split + early stop (Workflow §4.1 / §8.2).

roofline 的三重作用 (主文档 §5.1):
  ① 天花板  (peak FLOPS / bandwidth)
  ② 当前位置 (实测延迟)
  ③ 离天花板余量 + 撞哪面墙  ← 用于选 BD 坐标系 + 早停

`regime` 只用来选 BD 坐标系 (memory_bound -> 坐标系A, compute_bound -> B). 它在
"问题级" (算子+shape+硬件, 按 naive 实现) 判定并锁定; 个别 kernel 在搜索中漂过
ridge 属正常 (Workflow §4.1 note).

Device peaks are optional: when unknown, regime is still decided from AI vs a
default ridge, and the roofline early-stop simply never fires (min_latency=None).
"""

from __future__ import annotations

from dataclasses import dataclass

MEMORY_BOUND = "memory_bound"
COMPUTE_BOUND = "compute_bound"


@dataclass
class OperatorProfile:
    """Naive-implementation cost estimate of the operator (problem-level)."""
    flops: float            # total floating-point ops
    bytes: float            # total bytes moved (in + out, naive)

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else 0.0


@dataclass
class DeviceRoofline:
    """Hardware ceilings. Peaks optional (None disables the latency floor)."""
    peak_flops: float | None = None         # FLOP/s
    peak_bw_bytes_s: float | None = None     # bytes/s
    default_ridge: float = 8.0               # FLOP/byte, used when peaks unknown

    @property
    def ridge(self) -> float:
        if self.peak_flops and self.peak_bw_bytes_s:
            return self.peak_flops / self.peak_bw_bytes_s
        return self.default_ridge


@dataclass
class RooflineResult:
    arithmetic_intensity: float
    ridge: float
    regime: str
    min_latency_ms: float | None     # theoretical lower bound (None if peaks unknown)

    def early_stop_ok(self, best_latency_ms: float | None, eps: float = 0.05) -> bool:
        """True when best latency is within eps of the roofline floor (§8.2)."""
        if self.min_latency_ms is None or best_latency_ms is None:
            return False
        return best_latency_ms <= self.min_latency_ms * (1.0 + eps)


def diagnose(op: OperatorProfile, dev: DeviceRoofline | None = None) -> RooflineResult:
    dev = dev or DeviceRoofline()
    ai = op.arithmetic_intensity
    regime = MEMORY_BOUND if ai < dev.ridge else COMPUTE_BOUND
    min_latency_ms: float | None = None
    if dev.peak_flops and dev.peak_bw_bytes_s:
        # latency floor = max(compute time, memory time)
        t_compute = op.flops / dev.peak_flops
        t_memory = op.bytes / dev.peak_bw_bytes_s
        min_latency_ms = max(t_compute, t_memory) * 1000.0
    return RooflineResult(arithmetic_intensity=ai, ridge=dev.ridge,
                          regime=regime, min_latency_ms=min_latency_ms)


# --- op-family regime heuristic (used until a real roofline is wired in) --------
# High arithmetic-intensity families (many FLOP per byte -> compute_bound) vs
# low-intensity families (streaming / reduction / gather -> memory_bound). Keyed
# on substrings of the op/task name OR its ncnn layer type. Convolution/GEMM/
# MatMul/Deconv/attention/recurrent/linalg are compute_bound; everything else
# (elementwise, activation, pooling, reduction, norm, softmax, layout/index ops)
# is memory_bound. Unknown -> compute_bound default (per project decision: the
# compute_bound coordinate's algo_family axis is the richer search space, and a
# genuinely heavy op mis-labelled memory_bound loses its algorithmic variants).
_COMPUTE_KEYS = (
    "conv", "deconv", "convtranspose", "gemm", "matmul", "innerproduct",
    "linear", "winograd", "strassen", "im2col", "attention", "lstm", "gru",
    "rnn", "einsum", "det", "linalg", "bmm", "dense",
)
_MEMORY_KEYS = (
    "unary", "binary", "elementwise", "relu", "sigmoid", "tanh", "gelu",
    "softmax", "softplus", "hardsigmoid", "hardswish", "celu", "elu", "prelu",
    "clip", "abs", "exp", "log", "floor", "ceil", "round", "sign", "neg",
    "sqrt", "reciprocal", "sin", "cos", "tan", "add", "sub", "mul", "div",
    "pool", "reduce", "reduction", "argmax", "argmin", "cumsum", "cumulative",
    "norm", "batchnorm", "layernorm", "instancenorm", "groupnorm",
    "concat", "slice", "crop", "reshape", "permute", "transpose", "pad",
    "gather", "scatter", "index", "topk", "sort", "where", "grid", "sample",
    "depthtospace", "pixelshuffle", "upsample", "interp", "resize",
)


def guess_regime(name: str, ncnn_layer: str = "", default: str = COMPUTE_BOUND) -> str:
    """Rough regime guess from the op/task name + ncnn layer type, for use until a
    real roofline (device peaks + per-op FLOP/byte) is available. compute_bound ->
    algo_family × mapping grid; memory_bound -> layout × tiling grid.

    Compute-bound families win when matched (conv/gemm/matmul/deconv/etc.), since a
    heavy op mis-labelled memory_bound would never explore its algorithmic axis.
    """
    nm = (name or "").lower()
    layer = (ncnn_layer or "").lower().strip()

    def _decide(blob: str) -> str | None:
        if any(k in blob for k in _COMPUTE_KEYS):
            # pure depthwise conv is memory-bound (grouped conv stays compute)
            if ("depthwise" in blob or "dwconv" in blob) and "group" not in nm:
                return MEMORY_BOUND
            return COMPUTE_BOUND
        if any(k in blob for k in _MEMORY_KEYS):
            return MEMORY_BOUND
        return None

    # the resolved ncnn layer type is the most reliable signal (e.g. an einsum
    # that lowers to a Reduction is memory-bound despite its name) -> check first.
    return (layer and _decide(layer)) or _decide(nm) or default


def estimate_operator_profile(model_py: str) -> OperatorProfile:
    """Rough naive-cost estimate from the PyTorch reference model.

    bytes = (sum input elems + output elems) * 4 ; flops ≈ output elems (treat as
    ~1 op/elem — fine for the elementwise/unary ops M1/M2 target; richer ops can
    pass an explicit OperatorProfile instead).
    """
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("ds_model_rl", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = (mod.Model(*init) if init else mod.Model()).eval()
    inputs = mod.get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    in_elems = sum(int(t.numel()) for t in inputs)
    out_elems = int(out.numel())
    return OperatorProfile(flops=float(out_elems),
                           bytes=float((in_elems + out_elems) * 4))
