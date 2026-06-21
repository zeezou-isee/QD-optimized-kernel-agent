"""best-first baseline (Workflow §7.5) — the control arm for "用不用 QD".

因为目标是少轮数, QD 不是无脑首选。 并行跑一个 **直接 argmin** 的对照: 贪心 best-first,
不维护档案、不保多样, 每轮只对当前最优做一次"优化"变异。 用 best-fitness@相同预算
对比 QD: 若 QD 更好且 argmin 来自非主流格子 → 多样性付费成功; 若 best-first 更少轮内
追平 → 该算子欺骗性弱, 用基线即可。 把"用不用 QD"变成数据驱动决策, 而非信仰。
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from schemas import BasinValue, MeasureSample, ParameterizedTemplate, materialize
from inner import ConstraintEngine, inner_search


class _EvaluatorLike(Protocol):
    def evaluate(self, template: ParameterizedTemplate, point: dict[str, Any]) -> MeasureSample: ...


# vary the CURRENT BEST template toward "faster" (no diversity, no niches).
VaryTemplateFn = Callable[[ParameterizedTemplate, list], ParameterizedTemplate]


def run_best_first(
    *,
    baseline_template: ParameterizedTemplate,
    baseline_latency: float,
    evaluator: _EvaluatorLike,
    engine: ConstraintEngine,
    vary_fn: VaryTemplateFn,
    budget: int = 80,
    inner_budget: int = 8,
    patience: int = 4,
) -> dict:
    best_t = baseline_template
    best_lat = baseline_latency
    best_kernel = dict(baseline_template.kernel_files)
    rounds = 0
    stale = 0
    iters: list[dict] = []
    stopped = ""

    while rounds < budget:
        try:
            template = vary_fn(best_t, iters)
        except Exception as exc:  # noqa: BLE001
            stopped = f"vary_failed: {exc}"
            break
        basin: BasinValue = inner_search(template, evaluator, engine, budget=inner_budget)
        rounds += basin.n_evaluated
        improved = basin.correct and basin.best_latency_ms is not None and basin.best_latency_ms < best_lat
        if improved:
            best_lat = basin.best_latency_ms
            best_t = template
            best_kernel = materialize(template, basin.best_params or {})
            stale = 0
        else:
            stale += 1
        iters.append({"round": len(iters), "cand_latency": basin.best_latency_ms,
                      "best_latency": best_lat, "improved": improved,
                      "evaluated": basin.n_evaluated})
        if stale >= patience:
            stopped = "converged (patience)"
            break
    if not stopped:
        stopped = f"budget ({budget}) reached"

    return {"best_latency_ms": best_lat, "best_kernel": best_kernel,
            "rounds": rounds, "stopped_reason": stopped, "iterations": iters}


def compare(qd: dict, bf: dict, baseline_cell: tuple | None = None,
            tol: float = 0.01) -> dict:
    """Decide whether QD's extra machinery paid off (§7.5).

    - QD wins when it's faster AND its argmin came from a non-mainstream niche
      (≠ baseline cell) → landscape was deceptive, diversity paid for itself.
    - best-first wins when it matches/beats QD in fewer measurement rounds →
      operator is weakly deceptive, the baseline suffices.
    """
    qd_lat = qd.get("best_latency_ms")
    bf_lat = bf.get("best_latency_ms")
    qd_cell = tuple(qd.get("grid_argmin_cell") or ()) or None
    from_nonmainstream = bool(baseline_cell and qd_cell and qd_cell != tuple(baseline_cell))

    if qd_lat is None or bf_lat is None:
        verdict, why = "inconclusive", "missing latency from one arm"
    elif qd_lat < bf_lat * (1 - tol):
        verdict = "qd"
        why = ("QD faster" + (" via non-mainstream niche (diversity paid off)"
                              if from_nonmainstream else " (modest edge)"))
    elif bf_lat < qd_lat * (1 - tol):
        verdict, why = "best_first", "best-first faster — weakly deceptive operator"
    elif bf.get("rounds", 1e9) < qd.get("rounds", 1e9):
        verdict, why = "best_first", "tie on latency, best-first used fewer rounds"
    else:
        verdict, why = "tie", "comparable latency and rounds"

    return {"verdict": verdict, "reason": why,
            "qd_latency": qd_lat, "best_first_latency": bf_lat,
            "qd_rounds": qd.get("rounds"), "best_first_rounds": bf.get("rounds"),
            "qd_argmin_cell": list(qd_cell) if qd_cell else None,
            "argmin_from_nonmainstream": from_nonmainstream}
