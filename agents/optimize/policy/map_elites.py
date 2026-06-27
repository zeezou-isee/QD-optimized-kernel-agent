"""Outer MAP-Elites loop (Workflow §7) — the QD Policy.

质量偏置选亲代 → LLM 变异/重组 → 解析预筛(免实测) → 内层求 basin → 同 cell 竞争
→ roofline/预算/收敛 早停。 QD 当**手段**, 最终只报 argmin (§8.3)。

Cold start (§7.1): seed the archive from the experience pool (floor, 不过滤) + the
baseline; while coverage is low, the directive is 'diversify' (铺开优先于优化);
once enough niches are filled it switches to 'optimize' (quality-biased).

The proposer is injected as `vary_fn(parent_elite, directive, history) -> template`
so this loop is testable with a stub and reused by the real LLM proposer.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol

from schemas import BasinValue, MeasureSample, ParameterizedTemplate, materialize
from inner import ConstraintEngine, inner_search
from .archive import Archive, Elite
from .bd import classify
from .roofline import RooflineResult


class _EvaluatorLike(Protocol):
    def evaluate(self, template: ParameterizedTemplate, point: dict[str, Any]) -> MeasureSample: ...


VaryFn = Callable[[Elite, str, list], ParameterizedTemplate]


def _signature(t: ParameterizedTemplate) -> str:
    # identity = code + param space + structural intent (techniques/constraints).
    # Two templates with identical code but different declared optimization intent
    # are NOT duplicates (they target different niches).
    parts = [f"{k}={v}" for k, v in sorted(t.kernel_files.items())]
    parts += [f"{n}:{sorted(map(str, s.values))}" for n, s in sorted(t.params.items())]
    parts.append("tech=" + ",".join(sorted(t.techniques or [])))
    parts.append("cons=" + ",".join(sorted(t.constraints or [])))
    return "|".join(parts)


def _baseline_elite(baseline_template: ParameterizedTemplate, latency: float,
                    regime: str) -> Elite:
    return Elite(cell=classify(baseline_template.techniques, regime),
                 latency_ms=latency, kernel_code=dict(baseline_template.kernel_files),
                 params={}, techniques=list(baseline_template.techniques), source="seed")


def run_map_elites(
    *,
    baseline_template: ParameterizedTemplate,
    baseline_latency: float,
    evaluator: _EvaluatorLike,
    engine: ConstraintEngine,
    vary_fn: VaryFn,
    regime: str,
    roofline: RooflineResult | None = None,
    seeds: list[Elite] | None = None,
    archive: Archive | None = None,
    budget: int = 80,
    inner_budget: int = 8,
    coverage_target: int = 4,
    patience: int = 4,
    sigma: float = 0.0,
    rng_seed: int = 0,
) -> dict:
    rng = random.Random(rng_seed)
    arc = archive or Archive()

    # --- cold start: floor seeds (不过滤) + baseline ---
    for s in (seeds or []):
        arc.place(s, sigma=sigma)
    arc.place(_baseline_elite(baseline_template, baseline_latency, regime), sigma=sigma)

    best = arc.argmin()
    best_lat = best.latency_ms if best else baseline_latency
    rounds = 0
    stale = 0
    iters: list[dict] = []
    seen_sigs: set[str] = set()
    stopped = ""

    while rounds < budget:
        # roofline early stop (§8.2)
        if roofline and roofline.early_stop_ok(best_lat):
            stopped = "roofline_reached"
            break

        # directive: b铺开 first, optimize once enough niches are covered (§7.1)
        directive = "diversify" if arc.coverage() < coverage_target else "optimize"
        parent = arc.select_parents(1, rng=rng)[0]

        try:
            template = vary_fn(parent, directive, iters)
        except Exception as exc:  # noqa: BLE001
            iters.append({"round": len(iters), "error": f"vary_failed: {exc}"})
            stale += 1
            if stale >= patience:
                stopped = "vary_failed"
                break
            continue

        # analytic prefilter: dedup identical proposals before any measurement (§7.6)
        sig = _signature(template)
        if sig in seen_sigs:
            iters.append({"round": len(iters), "directive": directive,
                          "skipped": "duplicate proposal"})
            stale += 1
            if stale >= patience:
                stopped = "stalled (duplicates)"
                break
            continue
        seen_sigs.add(sig)

        # structural BD prelocation (known before the inner search, §4.3)
        cell = classify(template.techniques, regime)

        basin: BasinValue = inner_search(template, evaluator, engine, budget=inner_budget)
        rounds += basin.n_evaluated

        kept = False
        if basin.correct and basin.best_latency_ms is not None:
            elite = Elite(cell=cell, latency_ms=basin.best_latency_ms,
                          kernel_code=materialize(template, basin.best_params or {}),
                          params=basin.best_params or {},
                          techniques=list(template.techniques), source="search")
            kept = arc.place(elite, sigma=sigma)
            if basin.best_latency_ms < best_lat:
                best_lat = basin.best_latency_ms
                best = elite
                stale = 0
            else:
                stale += 1
        else:
            stale += 1

        iters.append({"round": len(iters), "directive": directive, "cell": list(cell),
                      "kept": kept, "cand_latency": basin.best_latency_ms,
                      "best_latency": best_lat, "coverage": arc.coverage(),
                      "evaluated": basin.n_evaluated, "pruned": basin.n_pruned})

        # convergence (§8.2): no global improvement for `patience` candidates
        if stale >= patience:
            stopped = "converged (patience)"
            break

    if not stopped:
        stopped = f"budget ({budget}) reached"

    best = arc.argmin()
    return {
        "best": best.to_dict() if best else None,
        "best_latency_ms": best.latency_ms if best else None,
        "best_kernel": best.kernel_code if best else dict(baseline_template.kernel_files),
        "coverage": arc.coverage(),
        "grid_argmin_cell": list(best.cell) if best else None,
        "rounds": rounds,
        "stopped_reason": stopped,
        "iterations": iters,
        "archive": arc.to_dict(),
        "regime": regime,
    }
