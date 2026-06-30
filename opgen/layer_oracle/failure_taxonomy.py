"""Failure taxonomy for kernel numeric verification — diagnosis-conditioned feedback.

Turns a single scalar `max_diff` into a LABELED, LOCALIZED, ACTIONABLE diagnostic so
the repair loop can condition on the failure MODE (not just "it's wrong"). Pure
numpy, deterministic, mutually-exclusive (priority order) — never an LLM guess.

`classify_failure(out, ref, tol) -> (category, detail)` covers the NUMERIC stage
only; the other stages are decided upstream:
  E0_GENERATE / E1_COMPILE / E2_RUNTIME_CRASH / E7_ISOLATION  (set elsewhere)

Numeric categories (checked in this priority order):
  E6_NUMERICAL_INSTABILITY  out has NaN/Inf
  E3_SHAPE_WRONG_COUNT      out.size != ref.size  (missing/extra axis, wrong reduce)
  E8_DISPATCH_COVERAGE      vulkan: output elements left as UNCHANGED INPUT (not processed)
  E4_LAYOUT_PERMUTED        same size, wrong shape, an axis permutation matches
  E5_VALUE_AFFINE           shape ok, out ≈ a*ref + b   (scale / offset / sign)
  E6_VALUE_NUMERICAL        shape ok, values wrong, no simple relation (algo/index)

Backend-aware extras (opts):
  input=<ncnn input>  -> E8 passthrough/coverage detection (vulkan dispatch bugs)
  backend="arm"       -> append NEON lane/tail error-pattern hints to E6
"""

from __future__ import annotations

import itertools

import numpy as np

_TOPK = 6
_AFFINE_MIN = 4   # don't claim scale/offset on tiny arrays (every scalar is an "offset")


def _shape(s) -> str:
    return "x".join(map(str, s)) if len(s) else "scalar"


def classify_failure(out, ref, tol: float = 2e-3, *,
                     input=None, backend: str = "base",
                     has_weights: bool = False) -> tuple[str, str]:
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    # E6a — non-finite output (numerical instability)
    nbad = int((~np.isfinite(out)).sum())
    if nbad:
        return ("E6_NUMERICAL_INSTABILITY",
                f"output has {nbad}/{out.size} non-finite (NaN/Inf) values -> numerical "
                f"instability (overflow / divide-by-zero / uninitialized memory). Check the "
                f"algorithm's numeric stability and that EVERY output element is written. "
                f"Per ncnn FAQ-produce-wrong-result: production-time default "
                f"`opt.use_fp16_packed=true` can overflow even when the LayerOracle baseline "
                f"(fp16 OFF) is clean — accumulate sums in fp32 inside your forward, even "
                f"on the arm/fp16 path, then cast to the output dtype only at write time.")

    # E3 — wrong number of elements
    if out.size != ref.size:
        ratio = out.size / max(ref.size, 1)
        return ("E3_SHAPE_WRONG_COUNT",
                f"WRONG ELEMENT COUNT: produced {out.size} ({_shape(out.shape)}) but expected "
                f"{ref.size} ({_shape(ref.shape)}); ratio={ratio:.3g}. You emit the wrong AMOUNT "
                f"of data — likely a missing/extra axis, a wrong reduction axis, or a "
                f"collapsed/expanded dim. Re-derive the output shape from the op semantics and "
                f"the ncnn (w,h,c) layout (batch dim dropped).")

    # E8 — vulkan dispatch coverage: output elements left as UNCHANGED INPUT (not processed)
    if backend == "vulkan" and input is not None:
        cov = _coverage(out, ref, np.asarray(input, dtype=np.float64), tol)
        if cov:
            return ("E8_DISPATCH_COVERAGE", cov)

    # E4 — same size, wrong shape, values are right but axes permuted (transpose)
    if out.shape != ref.shape and out.ndim == ref.ndim and 0 < out.ndim <= 4:
        for perm in itertools.permutations(range(out.ndim)):
            if tuple(np.array(out.shape)[list(perm)]) == tuple(ref.shape):
                if np.allclose(np.transpose(out, perm), ref, atol=tol, rtol=tol):
                    return ("E4_LAYOUT_PERMUTED",
                            f"OUTPUT IS A TRANSPOSE of the reference: your shape {_shape(out.shape)} "
                            f"vs ref {_shape(ref.shape)} — the VALUES are correct but the axes are "
                            f"permuted (apply axis permutation {perm}). You mapped the wrong torch "
                            f"axis onto ncnn w/h/c; fix the index mapping, not the math.")

    # values: compare on ref's shape (row-major)
    try:
        out_r = out.reshape(ref.shape)
    except ValueError:
        out_r = out.flatten().reshape(ref.shape)
    diff = np.abs(out_r - ref)

    # E5 — affine relation out ≈ a*ref + b
    if ref.size >= _AFFINE_MIN:
        det = _affine(out_r.flatten(), ref.flatten(), tol)
        if det:
            return ("E5_VALUE_AFFINE", det)

    # E6 — distributed value error, localized (+ arm NEON lane/tail hint)
    det = _localize(out_r, ref, diff, tol)
    if backend == "arm":
        lt = _lane_tail(ref, diff, tol)
        if lt:
            det += "\nNEON pattern: " + lt
    # Weight-misalignment signature: when the layer HAS bin weights and almost
    # every element is wrong with frequent sign flips, the usual cause is reading
    # a weight with the WRONG mb.load type (a primary/tagged weight read as type 1
    # consumes the 4-byte tag as the first float -> the whole buffer shifts by one
    # and the result looks like random sign-flipped noise). Surface the ncnn
    # contract so the repair round fixes the load type instead of guessing.
    if has_weights and out_r.size:
        wrong = float((diff > (tol + tol * np.abs(ref))).mean())
        both = (np.abs(out_r) > 1e-6) & (np.abs(ref) > 1e-6)
        signflip = float((np.sign(out_r[both]) != np.sign(ref[both])).mean()) if both.any() else 0.0
        if wrong > 0.8 and signflip > 0.3:
            det += ("\nWEIGHT-MISALIGNMENT SUSPECT: ~all values wrong with many sign "
                    "flips on a layer that loads weights. This is the classic "
                    "mb.load TYPE error — a PRIMARY weight (ncnn flag=0, tagged) "
                    "MUST be read with mb.load(size, 0); reading it with type 1 "
                    "eats the 4-byte tag and shifts every value. Check the REFERENCE "
                    "interface block: read each weight with TYPE = its `flag` "
                    "(0=primary/tagged, 1=secondary/raw). Mirror the built-in "
                    "layer's load_model exactly.")
    return ("E6_VALUE_NUMERICAL", det)


def _coverage(out: np.ndarray, ref: np.ndarray, inp: np.ndarray, tol: float) -> str | None:
    """vulkan: detect output elements that are UNCHANGED INPUT but wrong (= not processed).

    Robust even for ops where f(x)==x on part of the domain (abs on positives): we
    require BOTH out≈input AND out≉ref, so a correctly-passed-through element (which
    also matches ref) is NOT flagged.
    """
    o, r, i = out.flatten(), ref.flatten(), inp.flatten()
    if not (o.size == r.size == i.size):
        return None
    unchanged = np.abs(o - i) <= (tol + tol * np.abs(i))
    wrong = np.abs(o - r) > (tol + tol * np.abs(r))
    pt = unchanged & wrong
    n = int(pt.sum())
    if n == 0 or n / o.size < 0.02:
        return None
    first = int(np.argmax(pt))
    return (f"DISPATCH COVERAGE: {n}/{o.size} ({n / o.size:.1%}) output elements are UNCHANGED "
            f"INPUT but wrong (NOT processed; first unprocessed flat idx≈{first}). The compute "
            f"shader didn't cover all elements — your workgroup is multi-D while the dispatch is "
            f"1-D, or local_size/dispatcher mismatch. Use "
            f"set_optimal_local_size_xyz(subgroup_size,1,1) and dispatch dispatcher.w = total "
            f"element count.")


def _lane_tail(ref: np.ndarray, diff: np.ndarray, tol: float, elempack: int = 4) -> str:
    """arm: is the value error concentrated in the scalar tail or specific NEON lanes?"""
    d = diff.flatten()
    n = d.size
    if n < elempack:
        return ""
    wrong = d > (tol + tol * np.abs(ref.flatten()))
    if not wrong.any():
        return ""
    hints = []
    tail = n % elempack
    if tail and wrong[n - tail:].mean() > 0.8 and (n - tail == 0 or wrong[:n - tail].mean() < 0.2):
        hints.append(f"all errors are in the last {tail} (scalar-tail) elements -> fix the "
                     f"remainder loop after the `i+{elempack}<=size` vectorized part")
    lanes = [round(float(wrong[k::elempack].mean()), 2) for k in range(elempack)]
    if max(lanes) > 0.8 and min(lanes) < 0.2:
        hot = [k for k in range(elempack) if lanes[k] > 0.8]
        hints.append(f"errors concentrate in lane(s) {hot} (wrong-fraction by i%{elempack}={lanes}) "
                     f"-> a vld1q/vst1q lane indexing bug")
    return "; ".join(hints)


def _affine(of: np.ndarray, rf: np.ndarray, tol: float) -> str | None:
    if np.allclose(of, -rf, atol=tol, rtol=tol):
        return "output ≈ -reference (SIGN FLIP) — you negated the result somewhere."
    d = of - rf
    if np.allclose(d, d[0], atol=max(tol, 1e-4)) and abs(d[0]) > tol:
        return (f"output ≈ reference + {d[0]:.4g} (CONSTANT OFFSET) — likely a missing/extra "
                f"bias or eps term.")
    nz = np.abs(rf) > 1e-6
    if nz.any():
        k = of[nz] / rf[nz]
        if np.allclose(k, k[0], atol=max(tol, 1e-3)) and abs(k[0] - 1.0) > tol:
            return (f"output ≈ {k[0]:.4g} × reference (SCALE FACTOR) — likely a wrong scalar "
                    f"param (alpha/scale) or double/half counting.")
    A = np.vstack([rf, np.ones_like(rf)]).T
    try:
        (a, b), *_ = np.linalg.lstsq(A, of, rcond=None)
        if np.allclose(a * rf + b, of, atol=max(tol, 1e-3)) and (abs(a - 1) > tol or abs(b) > tol):
            return (f"output ≈ {a:.4g}×reference + {b:.4g} (AFFINE) — check scalar params "
                    f"(alpha/beta) and bias/eps.")
    except Exception:  # noqa: BLE001
        pass
    return None


def _localize(out_r: np.ndarray, ref: np.ndarray, diff: np.ndarray, tol: float,
              topk: int = _TOPK) -> str:
    md, mn = float(diff.max()), float(diff.mean())
    wrong = float((diff > (tol + tol * np.abs(ref))).mean())
    lines = [f"VALUE ERROR (algorithm/indexing): max_diff={md:.4g} mean_diff={mn:.4g} "
             f"wrong_frac={wrong:.1%} (no simple scale/offset/sign/transpose relation)."]
    d = np.asarray(diff)
    if d.ndim > 1:
        per = d.max(axis=tuple(range(1, d.ndim)))
        order = np.argsort(per)[::-1][:topk]
        lines.append("worst axis-0 indices (channel/row): "
                     + ", ".join(f"[{int(i)}]->{float(per[i]):.4g}" for i in order))
    of, rf, fl = out_r.flatten(), ref.flatten(), d.flatten()
    idx = np.argsort(fl)[::-1][:topk]
    lines.append("worst elements (flat idx: got vs expected): "
                 + ", ".join(f"{int(i)}: {of[i]:.4g} vs {rf[i]:.4g}" for i in idx))
    # ncnn-specific suspicion: if wrong_frac is very high and the output spatial
    # size (w*h*d, excluding channel) is NOT divisible by 4, the layer probably
    # cast the Mat to flat float* across channels and stepped through the
    # channel-gap (per FAQ-produce-wrong-result: "blob may have channel gap").
    if wrong > 0.5 and ref.ndim >= 3:
        spatial = 1
        for s in ref.shape[1:]:
            spatial *= int(s)
        if spatial % 4 != 0:
            lines.append(f"SUSPICION (channel gap): output spatial size "
                         f"{'*'.join(str(s) for s in ref.shape[1:])}={spatial} "
                         f"is NOT divisible by 4. ncnn pads `mat.cstep` per channel; "
                         f"if your forward casts the Mat to a flat float* and walks "
                         f"all c*h*w elements contiguously, it reads/writes the gap "
                         f"and corrupts channel boundaries. Use `mat.channel(q)` and "
                         f"iterate w*h*d elements PER channel; do NOT compute "
                         f"`(const float*)bottom_blob + offset` across channels.")
    return "\n".join(lines)
