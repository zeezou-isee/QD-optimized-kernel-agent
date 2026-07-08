"""inner_search — the inner parameter optimizer (Workflow §6).

Three stages on one parameterized template:
    ① analytic pruning   (免实测,砍非法点)         — ConstraintEngine
    ② coarse grid        (定位 basin)               — coarse_grid + Evaluator
    ③ local hill climb   (利用局部光滑微调)          — hill_climb + Evaluator

Returns the template's **basin value** = the best correct (params, latency) found
within the measurement budget, plus bookkeeping (n_evaluated/n_pruned/n_failed)
for the round-budget account (Workflow §8.1).

The evaluator is any object exposing `.evaluate(template, point) -> MeasureSample`
(the real CPU Evaluator, or a fake one in tests).
"""

from __future__ import annotations

from typing import Any, Protocol

from schemas import BasinValue, MeasureSample, ParameterizedTemplate
from .coarse_grid import coarse_points
from .constraint_engine import ConstraintEngine
from .hill_climb import hill_climb


class _EvaluatorLike(Protocol):
    def evaluate(self, template: ParameterizedTemplate, point: dict[str, Any]) -> MeasureSample: ...


def inner_search(
    template: ParameterizedTemplate,
    evaluator: _EvaluatorLike,
    engine: ConstraintEngine,
    *,
    budget: int = 10,
    coarse_per_axis: int = 3,
    coarse_max_points: int = 12,
) -> BasinValue:
    basin = BasinValue()
    seen: dict[tuple, MeasureSample] = {}

    def _key(p: dict[str, Any]) -> tuple:
        return tuple(sorted(p.items()))

    def measure(point: dict[str, Any], stage: str = "grid") -> float | None:
        """Feasibility → cache → real evaluate. Returns latency or None (unusable).
        Updates basin counters and the running best as a side effect. `stage` tags
        the sample ("grid" | "climb") for the paper-viz trace."""
        key = _key(point)
        if key in seen:
            s = seen[key]
            return s.latency_ms if s.correct else None
        # ① analytic pruning (免实测) — record which point + why, for the trace
        feas = engine.feasible(point, template.constraints)
        if not feas.ok:
            basin.n_pruned += 1
            basin.pruned.append({"point": dict(point), "reason": feas.reason, "stage": stage})
            return None
        if basin.n_evaluated >= budget:
            return None        # out of measurement budget
        sample = evaluator.evaluate(template, point)
        sample.stage = stage
        seen[key] = sample
        basin.samples.append(sample)
        basin.n_evaluated += 1
        if not sample.correct or sample.latency_ms is None:
            basin.n_failed += 1
            return None
        # track best
        if basin.best_latency_ms is None or sample.latency_ms < basin.best_latency_ms:
            basin.best_latency_ms = sample.latency_ms
            basin.best_params = dict(point)
            basin.best_sample = sample
            basin.correct = True
            basin.noise_floor_ms = sample.latency_std_ms
        return sample.latency_ms

    # ② coarse grid: locate the basin
    for pt in coarse_points(template.params, per_axis=coarse_per_axis,
                            max_points=coarse_max_points):
        if basin.n_evaluated >= budget:
            break
        measure(pt, "grid")

    # ③ hill climb from the best coarse point (if any), spending leftover budget
    if basin.best_params is not None and template.params:
        remaining = max(0, budget - basin.n_evaluated)
        if remaining > 0:
            best_pt, best_lat, _used = hill_climb(
                basin.best_params, basin.best_latency_ms, template.params,
                lambda p: measure(p, "climb"), budget=remaining)
            # measure() already folded improvements into basin via side effects.

    return basin
