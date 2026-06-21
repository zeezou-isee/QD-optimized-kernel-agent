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
