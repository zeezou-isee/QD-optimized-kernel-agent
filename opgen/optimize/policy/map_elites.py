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
from collections import Counter
from typing import Any, Callable, Protocol

from pathlib import Path

from schemas import BasinValue, MeasureSample, ParameterizedTemplate, materialize
from inner import ConstraintEngine, inner_search
from .archive import Archive, Elite
from .bd import classify, classify_with_novelty, posthoc_bd
from .roofline import RooflineResult
from . import sigma as _sigma


def _summarize_failures(basin: BasinValue) -> str:
    """One-line diagnosis of a basin's failed candidates, fed back to the proposer
    (closes the optimizer's feedback loop): dominant failure category + a sample."""
    fails = [s for s in basin.samples if not getattr(s, "correct", True)]
    if not fails:
        return ""
    cats = []
    for s in fails:
        cr = getattr(s, "correctness", None)
        if cr is not None and getattr(cr, "failure_category", ""):
            cats.append(cr.failure_category)
        elif "compile" in (s.error or ""):
            cats.append("E1_COMPILE")
        elif "crash" in (s.error or "") or "runtime" in (s.error or ""):
            cats.append("E2_RUNTIME_CRASH")
        else:
            cats.append(s.error or "other")
    top = Counter(cats).most_common(1)[0][0]
    rep = ""
    for s in fails:
        cr = getattr(s, "correctness", None)
        if cr is not None and getattr(cr, "failure_category", "") == top and cr.detail:
            rep = cr.detail
            break
    if not rep:
        for s in fails:
            rep = s.error or s.compile_log_tail or ""
            if rep:
                break
    return f"{len(fails)}/{len(basin.samples)} candidates failed; dominant={top}. e.g. {rep[:280]}"


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
    # --- axis-extension (Method M2.5.2): growable Σ ---
    backend: str = "base",
    wiki_root: Path | str | None = None,     # None => Σ read-only (synthesized fallback, no write-back)
    task_name: str = "",                     # for cross-task N_promote accounting
    n_promote: int = _sigma.DEFAULT_N_PROMOTE,
    record_trace: bool = False,              # persist per-round inner trajectory + pruned + param_space
) -> dict:
    rng = random.Random(rng_seed)
    arc = archive or Archive()

    # Σ registry for this backend. Loaded when a wiki_root is given; only then can
    # axis-extension persist a promotion back to disk (read-only ablation passes
    # wiki_root=None → novel labels still open niches in-run, just not written back).
    sg = _sigma.load(wiki_root, backend) if wiki_root is not None else None
    axis_extension_events: list[dict] = []   # promotions this run (Figure E data)
    novel_seen: list[dict] = []              # every novel-axis candidate that won a niche
    sigma_dirty = False                      # Σ mutated (pending counter or promotion) → persist

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

        # structural BD prelocation (known before the inner search, §4.3).
        # Σ-aware: an LLM-declared bd_label outside Σ opens a NOVEL niche and is
        # reported in `novel` (axis_name -> proposed value) for axis-extension.
        cell, novel = classify_with_novelty(
            template.techniques, regime, backend=backend,
            bd_labels=getattr(template, "bd_labels", None), wiki_root=wiki_root)

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

            # AXIS-EXTENSION WRITE-BACK (Method M2.5.2): a correct candidate that
            # WON/OPENED its cell (kept) AND carries a novel structural label
            # feeds Σ growth. record_win accumulates a cross-task counter; at
            # n_promote distinct tasks the value is promoted into Σ and persisted.
            if kept and novel:
                for axis_name, value in novel.items():
                    novel_seen.append({"axis": axis_name, "value": value,
                                       "cell": list(cell), "task": task_name})
                    if sg is not None:
                        ev = sg.record_win(regime, axis_name, value,
                                           task=task_name, n_promote=n_promote)
                        sigma_dirty = True   # pending counter advanced (or promoted)
                        if ev:
                            axis_extension_events.append(ev)
        else:
            stale += 1

        # post-hoc BD refinement (Method M2.3): if the best sample carries a
        # MEASURED micro-arch profile (on-device simpleperf path), derive a
        # refinement bin that sub-divides the niche. No-op in host search (no PMU).
        _bs = getattr(basin, "best_sample", None)
        _prof = getattr(_bs, "profile", None) if _bs is not None else None
        refine = posthoc_bd(_prof)

        rec = {"round": len(iters), "directive": directive, "cell": list(cell),
               "kept": kept, "novel": novel or None, "posthoc_refine": refine or None,
               "cand_latency": basin.best_latency_ms,
               "best_latency": best_lat, "coverage": arc.coverage(),
               "evaluated": basin.n_evaluated, "pruned": basin.n_pruned,
               "failure_summary": _summarize_failures(basin)}
        if record_trace:
            # full inner-search story for paper viz: the per-point climb trajectory
            # (grid then climb, in eval order), the analytically-pruned points +
            # reasons, and this template's parameter search space.
            rec["techniques"] = list(template.techniques or [])
            rec["param_space"] = {n: list(ps.values) for n, ps in (template.params or {}).items()}
            rec["trajectory"] = [
                {"point": sm.point, "latency_ms": sm.latency_ms, "correct": sm.correct,
                 "stage": sm.stage, "error": sm.error or None} for sm in basin.samples]
            rec["pruned_points"] = list(basin.pruned)
            rec["best_params"] = basin.best_params
        iters.append(rec)

        # convergence (§8.2): no global improvement for `patience` candidates
        if stale >= patience:
            stopped = "converged (patience)"
            break

    if not stopped:
        stopped = f"budget ({budget}) reached"

    # Persist Σ whenever a novel win mutated it — the `pending` cross-task
    # counter must survive between runs so a label can accumulate to n_promote
    # across DIFFERENT tasks, not just within one run. (No mutation → no write,
    # keeps the JSON churn-free; read-only ablation has sg=None → never writes.)
    if sg is not None and sigma_dirty:
        try:
            sg.save()
        except OSError:
            pass

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
        # axis-extension telemetry (Method M2.5.2 / Figure E)
        "axis_extension": {
            "promotions": axis_extension_events,       # values promoted INTO Σ this run
            "novel_niche_wins": novel_seen,            # every novel-axis niche win
            "sigma_size": sg.size() if sg is not None else None,
        },
    }
