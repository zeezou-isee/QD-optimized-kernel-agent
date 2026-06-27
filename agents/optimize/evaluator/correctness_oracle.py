"""CorrectnessOracle — the hard gate before any timing (Workflow §5.1).

先验正确,再谈快慢。 A candidate kernel that compiles, runs, and is *fast* but
computes the *wrong* answer must never enter the search — it would poison the
whole optimization line. This oracle compares a candidate's output against a
**reference** (the already-PyTorch-verified baseline kernel's output, 对拍) and
returns a pass/fail with the numeric diff.

Reference choice (plan line 42): we compare against the **baseline kernel**, not
PyTorch directly. The baseline was already verified == PyTorch by KernelAgent, so
对拍 baseline is equivalent and keeps inputs/params/weights byte-identical between
the two runs (no layout/precision drift in the comparison itself).
"""

from __future__ import annotations

import numpy as np

from schemas import CorrectnessReport


class CorrectnessOracle:
    """Holds the cached reference output; checks candidates against it."""

    def __init__(self, reference: np.ndarray, *, atol: float = 2e-3, rtol: float = 2e-3) -> None:
        # reference is the baseline kernel output in ncnn layout (batch dropped).
        self.reference = np.asarray(reference, dtype=np.float32)
        self.atol = atol
        self.rtol = rtol

    def check(self, candidate_out: np.ndarray) -> CorrectnessReport:
        out = np.asarray(candidate_out, dtype=np.float32)
        ref = self.reference
        # An optimization must not change the shape; reshape only flattens
        # benign layout differences (e.g. trailing singleton dims).
        try:
            out_r = out.reshape(ref.shape)
        except ValueError:
            return CorrectnessReport(
                passed=False,
                detail=f"shape mismatch: candidate {out.shape} vs reference {ref.shape}",
            )
        diff = np.abs(out_r - ref)
        max_diff = float(diff.max()) if diff.size else 0.0
        mean_diff = float(diff.mean()) if diff.size else 0.0
        passed = bool(np.allclose(out_r, ref, atol=self.atol, rtol=self.rtol))
        return CorrectnessReport(
            passed=passed, max_diff=max_diff, mean_diff=mean_diff,
            detail=f"max_diff={max_diff:.6g} mean_diff={mean_diff:.6g} "
                   f"atol={self.atol} rtol={self.rtol}",
        )
