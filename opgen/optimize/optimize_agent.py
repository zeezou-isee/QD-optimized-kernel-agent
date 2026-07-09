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
        model: str = "deepseek-v4-pro",
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
        n_promote: int = 3,                     # axis-extension: cross-task wins to grow Σ
        device_measure: bool = False,           # measure candidate+baseline latency on the REAL phone
        ncnn_py: str | Path | None = None,      # pnnx _ncnn.py: per-blob input squeeze policy
        record_trace: bool = False,             # persist per-round inner trajectory + pruned + bd_axes
        device_bench: int = 100,                # on-device --bench timed forwards
        device_warmup: int = 10,                # on-device --bench-warmup discarded forwards
        crossover_rate: float = 0.4,            # MAP-Elites P(crossover) per round (mutation=1-rate)
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
        self.n_promote = n_promote
        self.device_measure = bool(device_measure)
        self.ncnn_py = str(ncnn_py) if ncnn_py else None
        self.record_trace = bool(record_trace)
        self.device_bench = int(device_bench)
        self.device_warmup = int(device_warmup)
        self.crossover_rate = float(crossover_rate)

    # ------------------------------------------------------------------ dispatch
    @property
    def _rich(self) -> bool:
        return bool((self.model_py or self.evaluator_obj) and (self.llm_query or self.proposer))

    def run(self) -> OptimizeResult:
        if not self._rich:
            return self._run_legacy()
        # device gate: if on-device measurement is requested but no phone is
        # connected, ABORT before any LLM call — a device-less run silently falls
        # back to host wall-clock (fake ms-scale numbers) AND wastes tokens. Fail
        # fast with a clear reason instead.
        if self.device_measure:
            ok, why = self._device_ok()
            if not ok:
                raise RuntimeError(
                    f"device_measure is on but no usable phone device: {why}. "
                    f"Aborting BEFORE any LLM call to avoid wasting tokens. "
                    f"Connect a device (adb) or run with --device-verify off.")
        if self.policy == "map_elites":
            return self._run_map_elites()
        return self._run_rich()

    def _device_serial(self) -> str | None:
        """adb serial of the phone the data is measured on (recorded into the
        summary so the rollup filename reflects WHERE it was measured, not just
        whatever is plugged in at rollup time). None if unavailable."""
        if not self.device_measure:
            return None
        try:
            import subprocess
            out = subprocess.run(["adb", "get-serialno"], capture_output=True,
                                 text=True, timeout=15).stdout.strip()
            return out or None
        except Exception:  # noqa: BLE001
            return None

    def _device_ok(self) -> tuple[bool, str]:
        """Cheap preflight: is a usable phone (adb + NDK/lib) present? Mirrors the
        exact availability check the on-device measurer uses (no compile)."""
        try:
            from layer_oracle import DeviceOracle, VulkanDeviceOracle
            oracle = (VulkanDeviceOracle(ncnn_root=self.ncnn_root) if self.backend == "vulkan"
                      else DeviceOracle(ncnn_root=self.ncnn_root))
            return oracle.available()
        except Exception as exc:  # noqa: BLE001
            return False, f"device check failed: {exc}"

    # ------------------------------------------------------------------ rich M1
    def _run_rich(self) -> OptimizeResult:
        from evaluator import Evaluator
        from inner import ConstraintEngine, detect, inner_search

        hw_specs = self.hardware or detect()
        wiki = self._build_wiki()
        hw_extras = wiki.hardware_extras(self._hw_profile_key(hw_specs)) if wiki else {}
        engine = ConstraintEngine(hw_specs, extras=hw_extras)
        ev = self.evaluator_obj or Evaluator(
            baseline_kernel=self.baseline_kernel_code, model_py=self.model_py,
            ncnn_root=self.ncnn_root, workdir=self.workdir,
            weight_keys=self.weight_keys, params=self.params,
            warmup=self.warmup, runs=self.runs,
            backend=self.backend, base_files=self.base_files,
            device_measure=self.device_measure, ncnn_py=self.ncnn_py,
            device_bench=self.device_bench, device_warmup=self.device_warmup,
        )
        proposer = self.proposer or self._build_proposer(
            hw_specs.to_dict(), wiki=wiki, hw_extras=hw_extras,
        )

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

    def _build_proposer(self, hw_dict: dict[str, Any],
                        wiki: Any | None = None,
                        hw_extras: dict[str, float] | None = None,
                        regime: str | None = None):
        from proposer import LLMProposer
        merged_hw = dict(hw_dict)
        if hw_extras:
            merged_hw.update(hw_extras)
        # cache for downstream callers (e.g. _run_map_elites) that call this
        # without arguments today.
        if wiki is None and getattr(self, "_wiki_cached", None) is None:
            wiki = self._build_wiki()
            hw_extras = wiki.hardware_extras(self._hw_profile_key_from_dict(merged_hw)) if wiki else {}
            if hw_extras:
                merged_hw.update(hw_extras)
        # regime: caller (map_elites) may pass it; otherwise try a cheap static
        # estimate; default "unknown" (loader treats as mixed regime).
        if regime is None:
            regime = self.regime or self._infer_regime_static()
        return LLMProposer(
            task_name=self.task_name, baseline_kernel=self.baseline_kernel_code,
            hardware=merged_hw, llm_query=self.llm_query, model=self.model,
            backend=self.backend, wiki=wiki, regime=regime,
        )

    def _infer_regime_static(self) -> str:
        """Static regime guess for the M1 linear loop (which doesn't build a
        full RooflineResult). Falls back to 'unknown' on any error → the wiki
        treats 'unknown' as 'mixed' and injects both coordinate systems.
        """
        from policy.roofline import guess_regime
        fallback = guess_regime(self.op_class or self.task_name)
        # Only trust the AI-based roofline when real device peaks exist; otherwise
        # the naive profile always says memory_bound -> use the op-family heuristic.
        if not self.model_py or self.device_roofline is None:
            return fallback
        try:
            from policy.roofline import estimate_operator_profile, diagnose
            op_prof = self.operator_profile or estimate_operator_profile(self.model_py)
            rl = diagnose(op_prof, self.device_roofline)
            return rl.regime or fallback
        except Exception:  # noqa: BLE001
            return fallback

    # ------------------------------------------------------------------ wiki
    def _build_wiki(self):
        """Construct a WikiLoader once per run for the active backend. Returns
        None on 'base' backend (no wiki content targets 'base'), or when the
        environment variable KERNELGEN_WIKI is set to a disable value
        (`off`/`0`/`false`/`no`) — the A/B control arm. Cached on self.
        """
        cached = getattr(self, "_wiki_cached", "unset")
        if cached != "unset":
            return cached
        wiki = None
        import os
        disabled = os.environ.get("KERNELGEN_WIKI", "").strip().lower() in (
            "off", "0", "false", "no", "disabled",
        )
        if self.backend in ("arm", "vulkan") and not disabled:
            from proposer import WikiLoader
            # experience_pool/wiki/ lives at the repo root, i.e. 3 dirs above
            # this file (opgen/optimize/optimize_agent.py -> repo/experience_pool/wiki).
            wiki_root = Path(__file__).resolve().parents[2] / "experience_pool" / "wiki"
            wiki = WikiLoader(wiki_root, self.backend)
            if not wiki.enabled:
                wiki = None
        self._wiki_cached = wiki
        return wiki

    def _resolve_op_family(self) -> str:
        try:
            from ncnn_interface import guess_layer_from_task, layer_to_family
        except ImportError:
            return "unknown"
        layer = guess_layer_from_task(self.task_name)
        return layer_to_family(layer)

    @staticmethod
    def _hw_profile_key(hw_specs) -> str:
        """Map a HardwareSpecs instance to a wiki hardware-profile filename stem.
        v0: single Apple M5 profile per backend. Extend by inspecting arch /
        brand string as more profiles are added.
        """
        return "apple_m5"

    @staticmethod
    def _hw_profile_key_from_dict(hw: dict[str, Any]) -> str:
        return "apple_m5"

    # ------------------------------------------------------------- map-elites (M2/M3)
    def _run_map_elites(self) -> OptimizeResult:
        from evaluator import Evaluator
        from inner import ConstraintEngine, detect
        from schemas import ParameterizedTemplate
        from policy import (Archive, ExperiencePool, classify, diagnose,
                            estimate_operator_profile, run_map_elites,
                            run_best_first, compare)

        hw_specs = self.hardware or detect()
        wiki = self._build_wiki()
        hw_extras = wiki.hardware_extras(self._hw_profile_key(hw_specs)) if wiki else {}
        engine = ConstraintEngine(hw_specs, extras=hw_extras)
        ev = self.evaluator_obj or Evaluator(
            baseline_kernel=self.baseline_kernel_code, model_py=self.model_py,
            ncnn_root=self.ncnn_root, workdir=self.workdir,
            weight_keys=self.weight_keys, params=self.params,
            warmup=self.warmup, runs=self.runs,
            backend=self.backend, base_files=self.base_files,
            device_measure=self.device_measure, ncnn_py=self.ncnn_py,
            device_bench=self.device_bench, device_warmup=self.device_warmup,
        )
        # roofline diagnosis first so we can pass regime into the proposer
        # (wiki v1 keys BD-axis content by regime).
        op_prof = self.operator_profile
        if op_prof is None and self.model_py:
            op_prof = estimate_operator_profile(self.model_py)
        rl = diagnose(op_prof, self.device_roofline) if op_prof else None
        # Regime picks the BD coordinate system. Trust the roofline ONLY when we
        # have real device peaks; the naive estimate_operator_profile (≈1 FLOP/
        # elem) always yields memory_bound, which starved conv/gemm of their
        # algo_family grid. Without device peaks, use the op-family heuristic
        # (conv/gemm/matmul/... -> compute_bound). Replace once a real roofline
        # (device peaks + per-op FLOP/byte) is wired in.
        from policy import guess_regime
        if self.regime:
            regime = self.regime
        elif self.device_roofline is not None and rl is not None:
            regime = rl.regime
        else:
            regime = guess_regime(self.op_class or self.task_name)
        proposer = self.proposer or self._build_proposer(
            hw_specs.to_dict(), wiki=wiki, hw_extras=hw_extras, regime=regime,
        )

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

        import inspect as _inspect
        try:
            _prop_covered = "covered_cells" in _inspect.signature(proposer.vary).parameters
        except (ValueError, TypeError):
            _prop_covered = False

        def vary_fn(parent, directive, history, covered_cells=None):
            if _prop_covered:
                return proposer.vary(parent, directive, history, covered_cells=covered_cells)
            return proposer.vary(parent, directive, history)   # 3-arg proposer (e.g. test stub)

        def crossover_fn(a, b, history):
            return proposer.crossover(a, b, history)

        # axis-extension (Method M2.5.2): write-back only when wiki is ON —
        # `wiki is None` (KERNELGEN_WIKI=off ablation) keeps Σ read-only, so the
        # ablation arm neither reads nor grows the space.
        wiki_root = getattr(wiki, "wiki_root", None) if wiki else None
        me = run_map_elites(
            baseline_template=base_t, baseline_latency=base_lat, evaluator=ev,
            engine=engine, vary_fn=vary_fn, regime=regime, roofline=rl, seeds=seeds,
            budget=self.map_budget, inner_budget=self.inner_budget,
            coverage_target=self.coverage_target, patience=self.patience,
            backend=self.backend, wiki_root=wiki_root,
            task_name=self.op_class or self.task_name, n_promote=self.n_promote,
            record_trace=self.record_trace,
            crossover_fn=crossover_fn, crossover_rate=self.crossover_rate)

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

        best_lat = me.get("best_latency_ms")            # AVG (primary)
        improved = best_lat is not None and best_lat < base_lat
        res.best_kernel = me.get("best_kernel") or dict(self.baseline_kernel_code)
        _best_elite = me.get("best") or {}
        res.best_perf = {"avg": best_lat,
                         "min": _best_elite.get("latency_min_ms"),
                         "max": _best_elite.get("latency_max_ms")}
        res.best_round = 0 if improved else -1
        res.stopped_reason = me.get("stopped_reason", "")
        # BD grid definition (how the bins are partitioned) + inner-search config —
        # persisted so a paper figure can reconstruct the niche grid without re-
        # deriving it from the Σ registry.
        from policy import axes
        (a1n, a1v), (a2n, a2v) = axes(regime)
        bd_axes = {"axis1": {"name": a1n, "values": list(a1v)},
                   "axis2": {"name": a2n, "values": list(a2v)}}
        inner_config = {"map_budget": self.map_budget, "inner_budget": self.inner_budget,
                        "coverage_target": self.coverage_target, "patience": self.patience,
                        "coarse_per_axis": 3, "coarse_max_points": 12}
        res.extra = {"regime": regime, "roofline": rl.__dict__ if rl else None,
                     "coverage": me.get("coverage"), "rounds": me.get("rounds"),
                     "argmin_cell": me.get("grid_argmin_cell"),
                     "baseline_cell": list(baseline_cell), "baseline_latency_ms": base_lat,
                     "device_serial": self._device_serial() if self.device_measure else None,
                     "bd_axes": bd_axes, "inner_config": inner_config,
                     "archive": me.get("archive"), "iterations": me.get("iterations"),
                     "baseline_comparison": cmp,
                     # axis-extension telemetry (Method M2.5.2 / Figure E)
                     "axis_extension": me.get("axis_extension")}
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
        "max": getattr(sample, "latency_max_ms", None),
        "median": sample.latency_median_ms, "std": sample.latency_std_ms,
        "n_runs": sample.n_runs,
    }


def _rel(cand: float | None, baseline: float | None) -> float | None:
    if cand is None or baseline is None or baseline == 0:
        return None
    return (cand - baseline) / baseline
