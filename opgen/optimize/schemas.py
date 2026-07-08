"""Shared data contracts for the optimize agent (M1).

Design references:
  - 算子优化-问题建模与体系设计.md       (Proposer / Evaluator / Policy roles)
  - 算子优化-完整Workflow.md             (inner = analytical prune + coarse grid + hill climb)
  - 微观参数优化设计.md                  (LLM physical-constraint pruning + parameter spec)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    """One axis of the parameter space proposed by the LLM Proposer.

    `values` are explicit discrete candidates (the LLM hands these out together
    with the parameterized template — no need to learn the space online).
    Internally everything is treated as discrete (typical kernel knobs:
    TILE_M / UNROLL_K / VEC_WIDTH ...).
    """
    name: str
    values: list[Any]
    dtype: str = "int"           # "int" | "str"
    desc: str = ""

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"ParamSpec '{self.name}' has no candidate values")


# ---------------------------------------------------------------------------
# Parameterized template (the Proposer's output, the inner search's input)
# ---------------------------------------------------------------------------
@dataclass
class ParameterizedTemplate:
    """A kernel skeleton with <PARAM_NAME> placeholders + the param space.

    The Proposer emits one of these per optimization round. The inner search
    materializes specific points and asks the Evaluator to measure them.
    """
    kernel_files: dict[str, str]     # {basename: code with <PARAM_X> placeholders}
    params: dict[str, ParamSpec]     # {"TILE_M": ParamSpec(...), ...}
    class_name: str                  # ncnn layer class name (for LayerOracle)
    header: str                      # candidate header filename (cand_xxx.h)
    file: str                        # candidate cpp filename (cand_xxx.cpp)
    rationale: str = ""              # LLM-stated "why this should be faster"
    techniques: list[str] = field(default_factory=list)   # ["tiling","unroll","vectorize"]
    constraints: list[str] = field(default_factory=list)  # LLM-derived physical bounds
    # Explicit BD-axis labels the LLM declares for this proposal (Method M2.4:
    # "LLM as Σ→instance projector"). e.g. {"algo_family":"winograd",
    # "compute_mapping":"dotprod"}. A value OUTSIDE the current Σ vocabulary is a
    # deliberate axis-extension proposal (novel structural label) — see
    # policy/bd.classify_with_novelty + policy/sigma.record_win. Empty => fall
    # back to keyword classification from `techniques`.
    bd_labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = {k: asdict(v) for k, v in self.params.items()}
        return d


def materialize(template: ParameterizedTemplate, point: dict[str, Any]) -> dict[str, str]:
    """Replace <PARAM_NAME> placeholders with concrete values, return compilable
    {filename: code}. Missing point keys raise — every ParamSpec must be set.
    """
    missing = [k for k in template.params if k not in point]
    if missing:
        raise KeyError(f"materialize: missing params {missing}")
    out: dict[str, str] = {}
    for fname, code in template.kernel_files.items():
        for name, val in point.items():
            code = code.replace(f"<{name}>", str(val))
        out[fname] = code
    return out


# ---------------------------------------------------------------------------
# Evaluator output (one measurement on one (template, point))
# ---------------------------------------------------------------------------
@dataclass
class CorrectnessReport:
    passed: bool
    max_diff: float | None = None
    mean_diff: float | None = None
    detail: str = ""
    failure_category: str = ""   # diagnosis-conditioned label (shared failure taxonomy)


@dataclass
class MeasureSample:
    """The Evaluator's atomic output: one (template, point) measured once."""
    point: dict[str, Any]
    correct: bool
    latency_ms: float | None = None       # aggregated (avg on device / median or min on host) over N runs
    latency_min_ms: float | None = None
    latency_max_ms: float | None = None
    latency_median_ms: float | None = None
    latency_std_ms: float | None = None    # measurement noise floor for this point
    n_runs: int = 0
    correctness: CorrectnessReport | None = None
    compile_log_tail: str = ""             # short snippet for debugging
    error: str = ""                        # "compile failed" / "runtime crash" / ...
    stage: str = ""                        # inner-search stage: "grid" | "climb" (trace)


# ---------------------------------------------------------------------------
# Inner search output (basin value of one template)
# ---------------------------------------------------------------------------
@dataclass
class BasinValue:
    """The 'value' of one parameterized template = the best measurement found."""
    best_params: dict[str, Any] | None = None
    best_latency_ms: float | None = None
    best_sample: MeasureSample | None = None
    correct: bool = False                  # at least one point passed the oracle
    n_evaluated: int = 0                   # real measurements consumed
    n_pruned: int = 0                      # points dropped by analytical pruner
    n_failed: int = 0                      # compile/runtime failures during search
    samples: list[MeasureSample] = field(default_factory=list)
    noise_floor_ms: float | None = None    # σ estimate at best point
    pruned: list[dict[str, Any]] = field(default_factory=list)  # analytically-pruned {point, reason}


# ---------------------------------------------------------------------------
# Iteration / Result (consumed by OptimizeAgent.run() and operator_agent.py)
# Kept v0.3-compatible: OptimizeResult fields used downstream (best_round /
# best_kernel / best_perf / stopped_reason / iterations) are unchanged.
# ---------------------------------------------------------------------------
@dataclass
class OptimizeIteration:
    round_idx: int
    proposal_id: str
    techniques: list[str]
    basin: BasinValue
    kept: bool                             # this iteration became the new best
    delta_vs_best: float | None            # (cand - best) / best
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # BasinValue.samples already has MeasureSample dataclass; asdict handles it
        return d


@dataclass
class OptimizeResult:
    iterations: list[OptimizeIteration] = field(default_factory=list)
    best_round: int = -1                   # -1 = baseline still best
    best_kernel: dict[str, str] = field(default_factory=dict)
    best_perf: dict[str, Any] = field(default_factory=dict)  # {"avg": ms, "min": .., "max": ..}
    stopped_reason: str = ""
    policy: str = "linear"                  # "linear" (M1) | "map_elites" (M2/M3)
    extra: dict[str, Any] = field(default_factory=dict)   # archive / roofline / baseline-compare

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [i.to_dict() for i in self.iterations],
            "best_round": self.best_round,
            "best_kernel": self.best_kernel,
            "best_perf": self.best_perf,
            "stopped_reason": self.stopped_reason,
            "policy": self.policy,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Type alias for the legacy evaluator callback (v0.3 contract; M1 keeps it
# but the M1 inner loop primarily uses its own measure_harness for speed).
# ---------------------------------------------------------------------------
EvaluatorFn = Callable[[dict[str, str]], dict[str, Any]]
