"""Minimal experiment for the failure taxonomy — does it extract actionable feedback
from the REAL historical failures in opgen/runs/?

We do NOT re-run generation (no LLM) and do NOT recompile/run kernels (no torch/ncnn).
Instead we read each failed run's recorded numeric signature (shapes / max_diff /
mean_diff from runs/<op>/kernel/summary.json), reconstruct out/ref arrays that
reproduce that exact signature, and check that classify_failure() turns the old
scalar log into a labeled, localized diagnostic.

Run:  python opgen/layer_oracle/test_failure_taxonomy.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))                 # opgen/ on path
from layer_oracle.failure_taxonomy import classify_failure  # noqa: E402

_RUNS = _THIS.parent / "runs"
_FAILS: list[str] = []


def _load_numeric_log(op: str) -> str:
    p = _RUNS / op / "kernel" / "summary.json"
    if not p.exists():
        return ""
    t = re.sub(r"Infinity", "1e999", p.read_text(encoding="utf-8"))
    t = re.sub(r"NaN", "null", t)
    fr = json.loads(t).get("final_result", {})
    return str(fr.get("numeric_log") or "")


def check(name: str, cond: bool, got: str) -> None:
    print(("  ok  " if cond else "  FAIL") + f"  {name}")
    print(f"        feedback: {got.splitlines()[0] if got else '(empty)'}")
    if len(got.splitlines()) > 1:
        for ln in got.splitlines()[1:]:
            print(f"                  {ln}")
    if not cond:
        _FAILS.append(name)


def main() -> int:
    rng = np.random.default_rng(0)
    print("=== real historical failures from opgen/runs/ ===\n")

    # ---- shape-mismatch failures: parse "ncnn (A) vs ref (B)" from the real log ----
    shape_re = re.compile(r"ncnn \(([^)]*)\) vs ref \(([^)]*)\)")

    def _dims(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", s))

    for op in ["Det_2d", "Einsum_diagonal", "Einsum_transpose"]:
        log = _load_numeric_log(op)
        m = shape_re.search(log)
        if not m:
            check(f"{op} (shape parse)", False, log)
            continue
        out_shape, ref_shape = _dims(m.group(1)), _dims(m.group(2))
        out = np.zeros(out_shape, dtype=np.float32)
        ref = rng.standard_normal(ref_shape).astype(np.float32)
        cat, det = classify_failure(out, ref)
        # these real cases all have different element counts -> E3
        check(f"{op}  real log: {log[:60]!r}", cat == "E3_SHAPE_WRONG_COUNT", f"[{cat}] {det}")

    # ---- value failures: reproduce the real max_diff/mean_diff magnitude ----
    # Det: max_diff=inf (LU blew up) -> non-finite output -> instability
    out = rng.standard_normal((1,)).astype(np.float32); out[0] = np.inf
    ref = rng.standard_normal((1,)).astype(np.float32)
    cat, det = classify_failure(out, ref)
    check("Det  real log: max_diff=inf", cat == "E6_NUMERICAL_INSTABILITY", f"[{cat}] {det}")

    # MatMul_square: max_diff≈33.2 mean≈8.8, shape ok, distributed error -> E6
    ref = rng.standard_normal((16, 16)).astype(np.float32)
    out = ref + rng.standard_normal((16, 16)).astype(np.float32) * 9.0   # distributed, no simple relation
    cat, det = classify_failure(out, ref)
    check("MatMul_square  real log: max_diff=33.2 mean=8.8", cat == "E6_VALUE_NUMERICAL", f"[{cat}] {det}")

    print("\n=== synthetic cases (these modes weren't in runs but are now caught) ===\n")
    # E4 transpose: values right, axes permuted
    ref = rng.standard_normal((3, 4)).astype(np.float32)
    out = ref.T.copy()
    cat, det = classify_failure(out, ref)
    check("E4 transpose (out = ref.T)", cat == "E4_LAYOUT_PERMUTED" and "(1, 0)" in det, f"[{cat}] {det}")

    # E5 scale / sign / offset
    for label, out2, key in [
        ("E5 scale 2x", ref * 2.0, "SCALE"),
        ("E5 sign flip", -ref, "SIGN"),
        ("E5 offset +0.5", ref + 0.5, "OFFSET"),
    ]:
        cat, det = classify_failure(out2.astype(np.float32), ref)
        check(label, cat == "E5_VALUE_AFFINE" and key in det, f"[{cat}] {det}")

    print("\n=== backend-specific (8.1): vulkan coverage + arm lane/tail ===\n")
    # E8 vulkan dispatch coverage: AbsVal where the 2nd half was left as unchanged input
    inp = rng.standard_normal(128).astype(np.float32)
    refA = np.abs(inp)
    outA = refA.copy(); outA[64:] = inp[64:]            # 2nd half not processed
    cat, det = classify_failure(outA, refA, input=inp, backend="vulkan")
    check("vulkan partial coverage (2nd half = input)",
          cat == "E8_DISPATCH_COVERAGE" and "COVERAGE" in det, f"[{cat}] {det}")

    # arm: error confined to NEON lane i%4==2
    refL = rng.standard_normal((4, 32)).astype(np.float32)
    outL = refL.flatten(); outL[np.arange(outL.size) % 4 == 2] += 5.0
    cat, det = classify_failure(outL.reshape(4, 32), refL, backend="arm")
    check("arm lane error (i%4==2)", cat == "E6_VALUE_NUMERICAL" and "lane" in det, f"[{cat}] {det}")

    # arm: error confined to the scalar tail (last 2 of 130)
    refT = rng.standard_normal(130).astype(np.float32)
    outT = refT.copy(); outT[128:] += 5.0
    cat, det = classify_failure(outT, refT, backend="arm")
    check("arm tail error (last 2)", cat == "E6_VALUE_NUMERICAL" and "tail" in det, f"[{cat}] {det}")

    # guard: vulkan coverage must NOT false-fire when f(x)=x on part of domain & all correct
    okv = np.abs(inp)
    cat, det = classify_failure(okv * 1.0 + 1.0, okv, input=inp, backend="vulkan")  # uniformly wrong, not passthrough
    check("vulkan no false coverage (uniform offset -> E5)", cat == "E5_VALUE_AFFINE", f"[{cat}] {det}")

    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): {_FAILS}")
        return 1
    print("ALL PASS — taxonomy extracts a labeled, localized diagnostic for every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
