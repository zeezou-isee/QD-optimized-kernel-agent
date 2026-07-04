"""ConstraintEngine — analytic pruning (Workflow §4.4 / §6.1 step ①).

不让 Agent 盲猜参数,而是让它写出**物理约束方程**;本引擎在真机实测前用这些
方程把非法点(超 cache / 寄存器溢出 / 不整除 / 不对齐)砍掉,免实测。

Two sources of constraints, both applied:
  1. **LLM-derived equations** carried on the template (`constraints: list[str]`),
     e.g. "TILE_M*TILE_N*4 <= L1" or "UNROLL_K <= 8". Evaluated with a tiny safe
     arithmetic/comparison interpreter (no `eval`, no attribute/call access).
  2. **Built-in name-pattern heuristics** (defensive defaults) keyed on common
     knob names: UNROLL* ≤ register budget; VEC*/PACK* ∈ {1,2,4,(8)}.

The hardware namespace (L1, L2, VEC_BITS, FP32_PER_VEC, VECTOR_REGS) is injected
so equations can reference physical limits symbolically.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

from .hardware_specs import HardwareSpecs

# ---------------------------------------------------------------------------
# safe expression evaluator: arithmetic + comparison + boolean only
# ---------------------------------------------------------------------------
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_BOOLOPS = {ast.And: all, ast.Or: any}


class _Unsafe(Exception):
    pass


def _ev(node: ast.AST, names: dict[str, float]):
    if isinstance(node, ast.Expression):
        return _ev(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise _Unsafe(f"non-numeric constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise _Unsafe(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ev(node.left, names), _ev(node.right, names))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, names)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BoolOp) and type(node.op) in _BOOLOPS:
        vals = [bool(_ev(v, names)) for v in node.values]
        return _BOOLOPS[type(node.op)](vals)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, names)
        for op, comp in zip(node.ops, node.comparators):
            if type(op) not in _CMPOPS:
                raise _Unsafe(f"unsupported comparison {type(op).__name__}")
            right = _ev(comp, names)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    raise _Unsafe(f"unsupported expression node {type(node).__name__}")


def safe_eval(expr: str, names: dict[str, float]):
    """Evaluate an arithmetic/comparison expression with `names` bound. Raises
    _Unsafe on anything outside the whitelisted grammar."""
    tree = ast.parse(expr, mode="eval")
    return _ev(tree, names)


# ---------------------------------------------------------------------------
@dataclass
class FeasibilityReport:
    ok: bool
    reason: str = ""


class ConstraintEngine:
    def __init__(self, hw: HardwareSpecs, extras: dict[str, float] | None = None) -> None:
        """`extras` are backend-scoped hardware symbols supplied by WikiLoader
        (arm: L3/CACHE_LINE/HAS_DOTPROD/…; vulkan: SUBGROUP_SIZE/MAX_SHARED_MEM_BYTES/
        HAS_FP16/…). They merge on TOP of the CPU defaults so the LLM can reference
        the same symbols it saw in the prompt's hardware block. Silent skip on
        unknown symbols still applies (safe_eval catches _Unsafe) — the extras
        just widen what the LLM is allowed to actually resolve.
        """
        self.hw = hw
        # hardware constants available to LLM equations.
        self.hw_ns: dict[str, float] = {
            "L1": hw.l1d_bytes, "L1D": hw.l1d_bytes, "L2": hw.l2_bytes,
            "VEC_BITS": hw.vector_bits, "FP32_PER_VEC": hw.fp32_per_vector,
            "VECTOR_REGS": hw.vector_regs, "NEON": 1 if hw.arch.startswith(("arm", "aarch")) else 0,
        }
        if extras:
            for k, v in extras.items():
                if isinstance(v, (int, float, bool)):
                    self.hw_ns[k] = float(v)

    def feasible(self, point: dict, constraints: list[str] | None = None) -> FeasibilityReport:
        """True if `point` satisfies all LLM equations + built-in heuristics."""
        ns = {**self.hw_ns, **{k: v for k, v in point.items() if isinstance(v, (int, float))}}

        # 1) LLM-derived physical equations
        for expr in (constraints or []):
            expr = expr.strip()
            if not expr:
                continue
            try:
                ok = bool(safe_eval(expr, ns))
            except _Unsafe:
                # un-evaluatable (references a name we don't have / odd syntax):
                # skip rather than wrongly prune — be conservative.
                continue
            except Exception:  # noqa: BLE001
                continue
            if not ok:
                return FeasibilityReport(False, f"violates constraint: {expr}")

        # 2) built-in defensive heuristics on common knob names
        for name, val in point.items():
            if not isinstance(val, (int, float)):
                continue
            up = name.upper()
            if "UNROLL" in up and val > self.hw.vector_regs:
                return FeasibilityReport(False, f"{name}={val} > register budget "
                                                f"{self.hw.vector_regs} (spill guard)")
            if ("VEC" in up or "PACK" in up or "SIMD" in up) and val not in (1, 2, 4, 8, 16):
                return FeasibilityReport(False, f"{name}={val} not a valid SIMD width")
            if val <= 0:
                return FeasibilityReport(False, f"{name}={val} must be positive")
        return FeasibilityReport(True)
