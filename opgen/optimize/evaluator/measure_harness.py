"""MeasureHarness — turn a compiled kernel binary into a stable latency number.

Design ref: 算子优化-完整Workflow.md §5.3 (latency 不是干净标量).
M1 scope: warmup + N repeats + aggregate (min / median / std as the noise floor).
We do NOT recompile per repeat — the harness invokes the *cached* runner binary
(via CpuRunner.run_once) directly, so the only per-run cost is the forward pass
plus subprocess startup (an accepted M1 limitation, see cpu_runner.run_once).

控温/锁频/绑核 (Workflow §5.3) are explicitly out of M1 scope (see plan 不做清单).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # avoid import cycle at runtime
    from .cpu_runner import CpuRunner, RunArtifacts


@dataclass
class LatencyStats:
    """Aggregated timing over N good runs."""
    latency_ms: float           # the headline number (median by default)
    min_ms: float
    median_ms: float
    std_ms: float               # measurement noise floor σ for this point
    n_runs: int
    raw_ms: list[float]


class MeasureHarness:
    """Repeat-and-aggregate timing on top of a CpuRunner.

    aggregate="median" (default) is robust to occasional scheduler hiccups;
    "min" is closer to the no-interference lower bound (Workflow §5.3 lets you
    pick either — min is more stable under power noise, median under outliers).
    """

    def __init__(self, runner: "CpuRunner", *, warmup: int = 3, runs: int = 20,
                 aggregate: str = "median", timeout_s: float = 30.0) -> None:
        if runs < 1:
            raise ValueError("runs must be >= 1")
        if aggregate not in ("median", "min"):
            raise ValueError("aggregate must be 'median' or 'min'")
        self.runner = runner
        self.warmup = max(0, warmup)
        self.runs = runs
        self.aggregate = aggregate
        self.timeout_s = timeout_s

    def measure(self, art: "RunArtifacts") -> LatencyStats:
        """Run warmup + N timed iterations of the cached binary; aggregate.

        Raises RuntimeError if any timed run fails (caller treats that as a
        runtime crash — a correct kernel must run deterministically).
        """
        # warmup: fill caches / page in the binary; results discarded.
        for _ in range(self.warmup):
            self.runner.run_once(art, timeout_s=self.timeout_s)

        samples: list[float] = []
        for i in range(self.runs):
            ok, ms, err = self.runner.run_once(art, timeout_s=self.timeout_s)
            if not ok:
                raise RuntimeError(f"timed run {i} failed: {err}")
            samples.append(ms)

        med = statistics.median(samples)
        mn = min(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        headline = med if self.aggregate == "median" else mn
        return LatencyStats(latency_ms=headline, min_ms=mn, median_ms=med,
                            std_ms=std, n_runs=len(samples), raw_ms=samples)
