"""OptimizeAgent — M1: real inner-loop kernel optimizer.

Architecture (Workflow §3 three roles, M1 scope):
    Proposer  = LLMProposer        (baseline kernel -> parameterized template)
    Evaluator = Evaluator          (materialize -> compile -> oracle -> measure)
    Policy    = linear loop + inner_search(analytic prune + coarse grid + climb)

> M2 will replace the linear Policy with MAP-Elites + roofline; M3 adds the
> experience pool + best-first baseline. M1 keeps the v0.3 public surface so the
> OperatorAgent orchestrator needs NO changes.

Two run modes, dispatched automatically by `run()`:

* **Rich mode** (standalone / CLI / tests): constructed with `model_py` +
  (`llm_query` or an injected `proposer`). Runs the real M1 loop using an
  internal CPU Evaluator (its own measure harness — faster & more precise than
  re-installing into ncnn each round).

* **Legacy mode** (OperatorAgent integration): constructed with only the
  `evaluator` callback (concrete kernel -> {functional_ok, perf}) and no model/
  LLM. There is no way to author templates without an LLM, so this degrades to
  the v0.3 behaviour: keep the baseline, report no improvement. Fully backward
  compatible — operator_agent.py is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from schemas import (
    BasinValue,
    OptimizeIteration,
    OptimizeResult,
    ParameterizedTemplate,
    materialize,
)


class OptimizeAgent:
    def __init__(
        self,
        *,
        task_name: str,
        baseline_kernel_code: dict[str, str],
        # --- legacy (OperatorAgent) ---
        evaluator: Callable[[dict[str, str]], dict[str, Any]] | None = None,
        baseline_perf: dict[str, Any] | None = None,
        max_rounds: int = 5,
        improve_tol: float = 0.02,
        # --- rich mode (standalone M1 real loop) ---
        model_py: str | Path | None = None,
        ncnn_root: str | Path | None = None,
        llm_query: Callable[[str, str], str] | None = None,
        model: str = "z-ai/glm-5.1",
        proposer: Any | None = None,            # injectable (stub) Proposer for tests
        evaluator_obj: Any | None = None,       # injectable (fake) Evaluator for tests
        weight_keys: list[str] | None = None,
        params: dict[int, Any] | None = None,
        backend: str = "base",                  # "base" | "arm" | "vulkan"
        base_files: dict[str, str] | None = None,   # arm/vulkan: verified base layer code
        hardware: Any | None = None,            # HardwareSpecs override
        inner_budget: int = 10,
        warmup: int = 3,
        runs: int = 20,
        workdir: str | Path | None = None,
        # --- M2/M3: outer policy ---
        policy: str = "linear",                 # "linear" (M1) | "map_elites" (M2/M3)
        map_budget: int = 80,
        coverage_target: int = 4,
        patience: int = 4,
        regime: str | None = None,              # override roofline regime
        operator_profile: Any | None = None,    # OperatorProfile override
        device_roofline: Any | None = None,     # DeviceRoofline (peaks for early stop)
        experience_pool_path: str | Path | None = None,   # 兵器谱 warm-start + persist
        run_baseline_comparison: bool = False,  # §7.5 best-first control arm
        op_class: str = "",
    ) -> None:
        self.task_name = task_name
        self.baseline_kernel_code = dict(baseline_kernel_code)
        self.evaluator_cb = evaluator
        self.baseline_perf = baseline_perf or {}
        self.max_rounds = max_rounds
        self.improve_tol = improve_tol

        self.model_py = str(model_py) if model_py else None
        self.ncnn_root = ncnn_root
        self.llm_query = llm_query
        self.model = model
        self.proposer = proposer
        self.evaluator_obj = evaluator_obj
        self.weight_keys = weight_keys or []
        self.params = params or {}
        self.backend = backend
        self.base_files = base_files or {}
        self.hardware = hardware
        self.inner_budget = inner_budget
        self.warmup = warmup
        self.runs = runs
        self.workdir = workdir
        self.policy = policy
        self.map_budget = map_budget
        self.coverage_target = coverage_target
        self.patience = patience
        self.regime = regime
        self.operator_profile = operator_profile
        self.device_roofline = device_roofline
        self.experience_pool_path = experience_pool_path
        self.run_baseline_comparison = run_baseline_comparison
        self.op_class = op_class

    # ------------------------------------------------------------------ dispatch
    @property
    def _rich(self) -> bool:
        return bool((self.model_py or self.evaluator_obj) and (self.llm_query or self.proposer))

    def run(self) -> OptimizeResult:
        if not self._rich:
            return self._run_legacy()
        if self.policy == "map_elites":
            return self._run_map_elites()
        return self._run_rich()

    # ------------------------------------------------------------------ rich M1
    def _run_rich(self) -> OptimizeResult:
        from evaluator import Evaluator
        from inner import ConstraintEngine, detect, inner_search

        hw_specs = self.hardware or detect()
        engine = ConstraintEngine(hw_specs)
        ev = self.evaluator_obj or Evaluator(
            baseline_kernel=self.baseline_kernel_code, model_py=self.model_py,
            ncnn_root=self.ncnn_root, workdir=self.workdir,
            weight_keys=self.weight_keys, params=self.params,
            warmup=self.warmup, runs=self.runs,
            backend=self.backend, base_files=self.base_files,
        )
        proposer = self.proposer or self._build_proposer(hw_specs.to_dict())

        res = OptimizeResult()
        # measure the baseline as the reference best (it always exists & is correct).
        base_t = ParameterizedTemplate(
            kernel_files=dict(self.baseline_kernel_code), params={},
            class_name=ev.class_name, header=ev.header, file=ev.file,
            rationale="baseline", techniques=[],
        )
        base_sample = ev.evaluate(base_t, {})
        if not base_sample.correct or base_sample.latency_ms is None:
            res.stopped_reason = f"baseline measurement failed: {base_sample.error}"
            return res
        best_lat = base_sample.latency_ms
        res.best_kernel = dict(self.baseline_kernel_code)
        res.best_perf = _perf(base_sample)
        res.best_round = -1                     # -1 == baseline still best

        for i in range(self.max_rounds):
            try:
                template = proposer.propose(res.iterations)
            except Exception as exc:  # noqa: BLE001
                res.stopped_reason = f"propose_failed: {exc}"
                break

            basin = inner_search(template, ev, engine, budget=self.inner_budget)
            cand_lat = basin.best_latency_ms
            delta = _rel(cand_lat, best_lat)
            kept = basin.correct and cand_lat is not None and cand_lat < best_lat
            if kept:
                best_lat = cand_lat
                res.best_kernel = materialize(template, basin.best_params)
                res.best_perf = _perf(basin.best_sample)
                res.best_round = i

            res.iterations.append(OptimizeIteration(
                round_idx=i, proposal_id=f"r{i}-{'/'.join(template.techniques) or 'noop'}",
                techniques=template.techniques, basin=basin, kept=kept,
                delta_vs_best=delta,
                detail=(f"cand={cand_lat} best={best_lat} "
                        f"evaluated={basin.n_evaluated} pruned={basin.n_pruned} "
                        f"failed={basin.n_failed}"),
            ))

            # convergence: a meaningful but tiny improvement -> stop (§8.2).
            if delta is not None and abs(delta) < self.improve_tol:
                res.stopped_reason = (f"converged (|delta|={abs(delta):.4f} "
                                      f"< tol={self.improve_tol})")
                return res

        if not res.stopped_reason:
            res.stopped_reason = f"max_rounds ({self.max_rounds}) reached"
        return res

    def _build_proposer(self, hw_dict: dict[str, Any]):
        from proposer import LLMProposer
        return LLMProposer(
            task_name=self.task_name, baseline_kernel=self.baseline_kernel_code,
            hardware=hw_dict, llm_query=self.llm_query, model=self.model,
        )

    # ------------------------------------------------------------- map-elites (M2/M3)
    def _run_map_elites(self) -> OptimizeResult:
        from evaluator import Evaluator
        from inner import ConstraintEngine, detect
        from schemas import ParameterizedTemplate
        from policy import (Archive, ExperiencePool, classify, diagnose,
                            estimate_operator_profile, run_map_elites,
                            run_best_first, compare)

        hw_specs = self.hardware or detect()
        engine = ConstraintEngine(hw_specs)
        ev = self.evaluator_obj or Evaluator(
            baseline_kernel=self.baseline_kernel_code, model_py=self.model_py,
            ncnn_root=self.ncnn_root, workdir=self.workdir,
            weight_keys=self.weight_keys, params=self.params,
            warmup=self.warmup, runs=self.runs,
            backend=self.backend, base_files=self.base_files,
        )
        proposer = self.proposer or self._build_proposer(hw_specs.to_dict())

        # roofline diagnosis -> regime (selects BD coordinate system)
        op_prof = self.operator_profile
        if op_prof is None and self.model_py:
            op_prof = estimate_operator_profile(self.model_py)
        rl = diagnose(op_prof, self.device_roofline) if op_prof else None
        regime = self.regime or (rl.regime if rl else "memory_bound")

        res = OptimizeResult(policy="map_elites")
        base_t = ParameterizedTemplate(
            kernel_files=dict(self.baseline_kernel_code), params={},
            class_name=ev.class_name, header=ev.header, file=ev.file,
            rationale="baseline", techniques=[])
        base_sample = ev.evaluate(base_t, {})
        if not base_sample.correct or base_sample.latency_ms is None:
            res.stopped_reason = f"baseline measurement failed: {base_sample.error}"
            return res
        base_lat = base_sample.latency_ms
        baseline_cell = classify(base_t.techniques, regime)

        # experience-pool warm start (兵器谱, 同 regime 不过滤)
        pool = ExperiencePool(self.experience_pool_path) if self.experience_pool_path else None
        seeds = pool.seeds_for(regime, hardware=hw_specs.arch) if pool else []

        def vary_fn(parent, directive, history):
            return proposer.vary(parent, directive, history)

        me = run_map_elites(
            baseline_template=base_t, baseline_latency=base_lat, evaluator=ev,
            engine=engine, vary_fn=vary_fn, regime=regime, roofline=rl, seeds=seeds,
            budget=self.map_budget, inner_budget=self.inner_budget,
            coverage_target=self.coverage_target, patience=self.patience)

        # optional best-first control arm (§7.5)
        cmp = None
        if self.run_baseline_comparison:
            bf = run_best_first(
                baseline_template=base_t, baseline_latency=base_lat, evaluator=ev,
                engine=engine, vary_fn=lambda t, h: proposer.vary(t, "optimize", h),
                budget=self.map_budget, inner_budget=self.inner_budget,
                patience=self.patience)
            cmp = compare(me, bf, baseline_cell=baseline_cell)

        # persist the archive back into the 兵器谱
        if pool:
            pool.add_archive(Archive.from_dict(me["archive"]), regime=regime,
                             op_class=self.op_class or ev.class_name, hardware=hw_specs.arch)
            pool.save()

        best_lat = me.get("best_latency_ms")
        improved = best_lat is not None and best_lat < base_lat
        res.best_kernel = me.get("best_kernel") or dict(self.baseline_kernel_code)
        res.best_perf = {"avg": best_lat, "min": best_lat}
        res.best_round = 0 if improved else -1
        res.stopped_reason = me.get("stopped_reason", "")
        res.extra = {"regime": regime, "roofline": rl.__dict__ if rl else None,
                     "coverage": me.get("coverage"), "rounds": me.get("rounds"),
                     "argmin_cell": me.get("grid_argmin_cell"),
                     "baseline_cell": list(baseline_cell), "baseline_latency_ms": base_lat,
                     "archive": me.get("archive"), "iterations": me.get("iterations"),
                     "baseline_comparison": cmp}
        return res

    # ------------------------------------------------------------------ legacy
    def _run_legacy(self) -> OptimizeResult:
        """v0.3-compatible degrade: no LLM/model => cannot author templates.

        Keep the baseline; if an external evaluator callback is available, run it
        once on the baseline so best_perf reflects a real benchmark.
        """
        res = OptimizeResult()
        res.best_kernel = dict(self.baseline_kernel_code)
        res.best_perf = dict(self.baseline_perf)
        res.best_round = -1
        if self.evaluator_cb is not None:
            try:
                ev = self.evaluator_cb(self.baseline_kernel_code)
                res.best_perf = ev.get("perf", {}) or res.best_perf
            except Exception as exc:  # noqa: BLE001
                res.stopped_reason = f"legacy evaluator crashed: {exc}"
                return res
        res.stopped_reason = ("legacy mode: no LLM proposer (M1 real loop needs "
                              "model_py + llm_query); baseline kept")
        return res


# ---------------------------------------------------------------------------
def _perf(sample) -> dict[str, Any]:
    return {
        "avg": sample.latency_ms, "min": sample.latency_min_ms,
        "median": sample.latency_median_ms, "std": sample.latency_std_ms,
        "n_runs": sample.n_runs,
    }


def _rel(cand: float | None, baseline: float | None) -> float | None:
    if cand is None or baseline is None or baseline == 0:
        return None
    return (cand - baseline) / baseline
