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
from layer_oracle import classify_failure   # shared diagnosis-conditioned taxonomy


class CorrectnessOracle:
    """Holds the cached reference output; checks candidates against it.

    On failure it emits a labeled, localized diagnostic via the shared failure
    taxonomy (same engine as KernelAgent), backend-aware (arm lane/tail hints,
    vulkan dispatch-coverage via `input`), so the optimizer's variation operator
    can be conditioned on WHY a candidate was wrong, not just that it was.
    """

    def __init__(self, reference: np.ndarray, *, atol: float = 2e-3, rtol: float = 2e-3,
                 backend: str = "base", input: np.ndarray | None = None) -> None:
        # reference is the baseline kernel output in ncnn layout (batch dropped).
        self.reference = np.asarray(reference, dtype=np.float32)
        self.atol = atol
        self.rtol = rtol
        self.backend = backend
        self.input = None if input is None else np.asarray(input, dtype=np.float32)

    def check(self, candidate_out: np.ndarray) -> CorrectnessReport:
        out = np.asarray(candidate_out, dtype=np.float32)
        ref = self.reference
        # An optimization must not change the shape; reshape only flattens benign
        # layout differences (e.g. trailing singleton dims).
        max_diff = mean_diff = None
        try:
            out_r = out.reshape(ref.shape)
            diff = np.abs(out_r - ref)
            max_diff = float(diff.max()) if diff.size else 0.0
            mean_diff = float(diff.mean()) if diff.size else 0.0
            if bool(np.allclose(out_r, ref, atol=self.atol, rtol=self.rtol)):
                return CorrectnessReport(
                    passed=True, max_diff=max_diff, mean_diff=mean_diff,
                    detail=f"max_diff={max_diff:.6g} atol={self.atol} rtol={self.rtol}")
        except ValueError:
            pass  # shape mismatch -> the taxonomy handles it (E3/E4)
        # failed: diagnosis-conditioned feedback
        cat, det = classify_failure(out, ref, self.atol, input=self.input, backend=self.backend)
        return CorrectnessReport(passed=False, max_diff=max_diff, mean_diff=mean_diff,
                                 detail=f"[{cat}] {det}", failure_category=cat)
