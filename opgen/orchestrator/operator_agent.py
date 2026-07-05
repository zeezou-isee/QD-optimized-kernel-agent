"""OperatorAgent: decision-driven end-to-end orchestrator for "add a new ncnn op".

Pipeline (driven by an existence check on ncnn):

  [1] KernelAgent          : write kernel + numeric vs PyTorch
  [2] BRIDGE               : install kernel + rebuild libncnn
  [3] existence check      : probe_pnnx_ir.baseline_supported
        [3a] YES (already in ncnn)        -> skip GraphAgent, use baseline IR
        [3b] NO                            -> GraphAgent (forced target, up to 15 rounds)
                                              fail -> abort
  [4] end_to_end_numeric   : Net runner vs PyTorch  (FUNCTIONAL)
  [5] PRODUCTION           : compile + correctness [+ on-device simpleperf profile]
  [6] (optional) OptimizeAgent: REAL two-layer kernel optimizer (LLM proposer +
        inner search / MAP-Elites QD) on the authored kernel. Each candidate is
        verified (对拍 baseline) + timed in a fast LayerOracle loop; the winner is
        re-installed and re-validated through production before being swapped into
        [7]. Runs for BOTH native and new operators (for native ops, the custom
        kernel is optimized even though the graph uses baseline IR).
  [7] cleanup / --install
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import KERNELGEN_ROOT, RUNS_ROOT, GraphConfig
from adapter_agent import AdapterAgent
from graph_agent import GraphAgent
from graph_pipeline import probe_pnnx_ir
from graph_schemas import write_json
from kernel_agent import KernelAgent
from layer_oracle import NetOracle, parse_ncnn_io, pnnx_driven_ncnn_inputs
from ncnn_tree_guard import arm_guard
from optimize_agent import OptimizeAgent


class OperatorAgent:
    def __init__(
        self,
        *,
        task_name: str,
        model_py: str | Path | None = None,
        model: str = "deepseek-v4-pro",
        max_rounds: int = 8,                  # KernelAgent default rounds
        graph_max_rounds: int = 15,           # GraphAgent rounds (raised from 8)
        ncnn_root: str | Path | None = None,
        dataset_root: str | Path | None = None,
        torch_install_dir: str | Path | None = None,
        end_to_end: bool = True,              # install + rebuild + net numeric (temporary)
        install: bool = False,                # PERMANENTLY register the verified op
        # production validation (MoKA-style; runs after end-to-end numeric)
        compile_mode: str = "build_lib",      # "build_lib" | "build_full"
        benchmark: bool = False,              # android benchncnn; auto-skip w/o device
        # optimization stage (real OptimizeAgent: M1 inner loop / M2-M3 MAP-Elites)
        optimize: bool = False,
        max_optimize_rounds: int = 5,
        improve_tol: float = 0.02,
        optimize_policy: str = "map_elites",  # "linear" (M1) | "map_elites" (M2/M3)
        optimize_map_budget: int = 60,
        optimize_inner_budget: int = 8,
        optimize_coverage_target: int = 4,
        experience_pool_path: str | Path | None = None,   # 兵器谱 warm-start + persist
        backends: list[str] | None = None,    # subset of ["base","arm"]; default ["base"]
        allow_backend_fallback: bool = False, # if a target backend (arm) fails: off=abort, on=base-only
        auto_cleanup_ncnn: bool = False,      # ncnn-tree guard: True => silently clean a dirty tree on startup;
                                              # False (default) => refuse to run on a dirty tree (safer)
        e2e_repair_max_attempts: int = 2,     # how many repair rounds to spend on e2e_repair after
                                              # a LayerOracle-passed kernel fails end-to-end NetOracle
        adapt_e2e: bool = True,               # use the contract-driven AdapterAgent for e2e_repair
                                              # (vs. the legacy guess-driven KernelAgent reseed)
        llm_query: Callable[[str, str], str] | None = None,
    ) -> None:
        self.task_name = task_name
        self.model_py = model_py
        self.model = model
        self.max_rounds = max_rounds
        self.graph_max_rounds = graph_max_rounds
        self.ncnn_root = Path(ncnn_root) if ncnn_root else GraphConfig().ncnn_root
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.torch_install_dir = Path(torch_install_dir) if torch_install_dir else None
        self.install = install
        # registering permanently requires the full end-to-end path
        self.end_to_end = end_to_end or install
        self.compile_mode = compile_mode
        self.do_benchmark = benchmark
        self.optimize = optimize
        self.max_optimize_rounds = max_optimize_rounds
        self.improve_tol = improve_tol
        self.optimize_policy = optimize_policy
        self.optimize_map_budget = optimize_map_budget
        self.optimize_inner_budget = optimize_inner_budget
        self.optimize_coverage_target = optimize_coverage_target
        self.experience_pool_path = experience_pool_path
        self.backends = backends or ["base"]
        self.want_arm = "arm" in self.backends
        self.allow_backend_fallback = allow_backend_fallback
        self.llm_query = llm_query
        self.auto_cleanup_ncnn = bool(auto_cleanup_ncnn)
        self.e2e_repair_max_attempts = max(0, int(e2e_repair_max_attempts))
        self.adapt_e2e = bool(adapt_e2e)
        # ncnn-tree guard is armed lazily in run() so __init__ stays non-throwing.
        self._ncnn_guard = None
        # baseline probe cache populated by _early_baseline_probe()
        self._cached_baseline = None
        # principal native ncnn layer type pnnx emits (set in run()); guards retarget
        self._force_analog = None

    # ------------------------------------------------------------------ paths
    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.task_name / "operator"

    def _cfg(self, run_numeric: bool, max_rounds: int | None = None) -> GraphConfig:
        return GraphConfig(
            ncnn_root=self.ncnn_root, dataset_root=self.dataset_root, model=self.model,
            max_rounds=max_rounds if max_rounds is not None else self.max_rounds,
            run_numeric=run_numeric, torch_install_dir=self.torch_install_dir,
        )

    def _resolve_model_py(self) -> str:
        if self.model_py:
            return str(self.model_py)
        base = self.dataset_root or (KERNELGEN_ROOT / "MobileKernelBench_git"
                                     / "dataset" / "Mobilekernelbench")
        return str(sorted(Path(base).rglob(f"{self.task_name}.py"))[0])

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        summary: dict = {"task_name": self.task_name, "phases": {}}

        # ---------------- [0] ncnn-tree guard (A+C) ----------------
        # (C) refuse to run on a dirty ncnn tree (or auto-clean if asked);
        # (A) take a snapshot SHA + register atexit/SIGTERM/SIGINT restore so any
        #     crash / kill chain still brings the tree back. Except --install,
        #     which intentionally keeps the injected pass permanent.
        if not self.install:
            self._ncnn_guard = arm_guard(self.ncnn_root, auto_cleanup=self.auto_cleanup_ncnn)

        # ---------------- [0b] Baseline probe (FIRST — feeds [1] and [3]) -----
        # Run pnnx ONCE up front to learn what ncnn layer type it actually emits
        # for this op. That layer type becomes a HARD CONSTRAINT on KernelAgent's
        # analog_layer choice — prevents the "LLM picks InnerProduct but pnnx
        # emits Gemm" class of e2e failures. The cached probe artifacts also
        # feed the later existence-check (no duplicate pnnx invocation).
        force_analog, _baseline = self._early_baseline_probe()
        # remember the principal native ncnn layer type pnnx emits for this op —
        # used to guard the output-layer retarget against DECOMPOSED ops (where the
        # output layer is a different native type, e.g. Gemm_alpha's final Add).
        self._force_analog = force_analog
        if force_analog:
            print(f"[orchestrator] baseline-probe: pnnx emits ncnn `{force_analog}` "
                  f"→ pinning KernelAgent analog_layer")
        else:
            print("[orchestrator] baseline-probe: no single ncnn layer detected — "
                  "KernelAgent will pick analog_layer itself")

        # ---------------- [1] Kernel ----------------
        print("\n===== [1] KernelAgent (kernel, numeric vs PyTorch) =====")
        kernel_sum = KernelAgent(task_name=self.task_name, model_py=self.model_py,
                                 cfg=self._cfg(run_numeric=True),
                                 llm_query=self.llm_query,
                                 force_analog_layer=force_analog).run()
        kernel_ok = kernel_sum.get("status") == "success"
        kprof = kernel_sum.get("kernel_profile") or {}
        kcode = (kernel_sum.get("final_result") or {}).get("response_code") or {}
        cls = kprof.get("class_name", "")
        summary["phases"]["kernel"] = {
            "status": kernel_sum.get("status"), "rounds": kernel_sum.get("rounds"),
            "max_diff": (kernel_sum.get("final_result") or {}).get("max_diff"),
            "class_name": cls,
        }
        print(f"[orchestrator] kernel: {kernel_sum.get('status')} (class={cls})")
        if not kernel_ok:
            summary["status"] = "fail"; summary["note"] = "kernel phase failed; aborting."
            write_json(self.run_dir / "summary.json", summary)
            return summary

        # ---------------- [1b] ARM kernel (optional) ----------------
        arm_code: dict = {}
        arm_ok = False
        if self.want_arm:
            print("\n===== [1b] KernelAgent (arm NEON kernel, numeric vs PyTorch) =====")
            arm_sum = KernelAgent(task_name=self.task_name, model_py=self.model_py,
                                  cfg=self._cfg(run_numeric=True), llm_query=self.llm_query,
                                  backend="arm", base_kernel_code=kcode, base_profile=kprof).run()
            arm_ok = arm_sum.get("status") == "success"
            arm_code = (arm_sum.get("final_result") or {}).get("response_code") or {}
            summary["phases"]["kernel_arm"] = {
                "status": arm_sum.get("status"), "rounds": arm_sum.get("rounds"),
                "max_diff": (arm_sum.get("final_result") or {}).get("max_diff"),
                "class_name": (arm_sum.get("kernel_profile") or {}).get("class_name"),
            }
            print(f"[orchestrator] arm kernel: {arm_sum.get('status')}")
            # A requested target backend is a hard gate by default: if arm fails the
            # whole operator run fails. Only when --allow-backend-fallback is set do
            # we degrade to base-only (arm as a non-blocking accelerator).
            if not arm_ok:
                if self.allow_backend_fallback:
                    arm_code = {}
                    print("[orchestrator] arm kernel failed — continuing with base only "
                          "(fallback enabled)")
                else:
                    summary["status"] = "fail"
                    summary["note"] = ("arm kernel failed and backend fallback is disabled "
                                       "(--allow-backend-fallback off); aborting.")
                    write_json(self.run_dir / "summary.json", summary)
                    print("[orchestrator] arm kernel failed — aborting "
                          "(fallback disabled; pass --allow-backend-fallback to degrade to base)")
                    return summary

        # ---------------- [2] Bridge ----------------
        netoc = NetOracle(ncnn_root=self.ncnn_root, workdir=RUNS_ROOT / "_net")
        handle = None
        arm_handle = None
        if self.end_to_end:
            print("\n===== [2] BRIDGE: install kernel(s) + rebuild libncnn =====")
            handle = netoc.install_layer(kcode, cls)                       # base -> src/layer/
            if arm_code:
                arm_handle = netoc.install_layer(arm_code, cls, subdir="arm", add_cmake=False)
            ok, log = netoc.rebuild_libncnn()
            (self.run_dir / "libncnn_rebuild.log").write_text(log, encoding="utf-8")
            summary["phases"]["install"] = {"installed": True, "libncnn_rebuilt": ok,
                                            "backends": ["base"] + (["arm"] if arm_code else [])}
            print(f"[orchestrator] install+rebuild libncnn: {'ok' if ok else 'FAILED'} "
                  f"(backends: base{'+arm' if arm_code else ''})")
            if not ok:
                netoc.restore(handle)
                if arm_handle: netoc.restore(arm_handle)
                netoc.rebuild_libncnn()
                summary["status"] = "fail"; summary["note"] = "libncnn rebuild failed."
                write_json(self.run_dir / "summary.json", summary)
                return summary

        graph_sum: dict = {}
        graph_ok = False
        already_in_ncnn = False
        e2e_ok = None
        prod_ok = None
        all_ok = False
        opt_result: dict | None = None
        try:
            # ------------- [3] Existence check (skip graph if already supported) -------------
            print("\n===== [3] Existence check (probe baseline pnnx) =====")
            already_in_ncnn, baseline_graph_sum = self._check_already_in_ncnn()
            summary["phases"]["existence_check"] = {
                "already_in_ncnn": already_in_ncnn,
                "baseline_op_types": baseline_graph_sum.get("_baseline_op_types"),
                "residual_aten": baseline_graph_sum.get("_residual_aten"),
            }
            print(f"[orchestrator] already_in_ncnn={already_in_ncnn} "
                  f"op_types={baseline_graph_sum.get('_baseline_op_types')}")

            if already_in_ncnn:
                # use baseline IR as if GraphAgent had produced it
                graph_sum = baseline_graph_sum
                graph_ok = True
                summary["phases"]["graph"] = {"status": "skipped (already supported by ncnn)",
                                              "rounds": 0, "forced_target": None}
            else:
                # ------------- [3b] GraphAgent (forced target) -------------
                print(f"\n===== [3b] GraphAgent (forced target={cls}, max_rounds={self.graph_max_rounds}) =====")
                graph_sum = GraphAgent(
                    task_name=self.task_name, model_py=self.model_py,
                    cfg=self._cfg(run_numeric=False, max_rounds=self.graph_max_rounds),
                    llm_query=self.llm_query, force_target_layer=cls,
                ).run()
                graph_ok = graph_sum.get("status") == "success"
                summary["phases"]["graph"] = {
                    "status": graph_sum.get("status"), "rounds": graph_sum.get("rounds"),
                    "forced_target": cls,
                }
                print(f"[orchestrator] graph: {graph_sum.get('status')}")
                if not graph_ok:
                    # abort: graph did not converge -> can't validate or optimize
                    summary["status"] = "fail"
                    summary["note"] = f"graph did not converge in {self.graph_max_rounds} rounds; aborting."
                    return summary  # finally still runs cleanup

            # ------------- [4] End-to-end numeric (functional) -------------
            if self.end_to_end:
                num = self._net_numeric(netoc, graph_sum, op_class=cls)
                summary["phases"]["end_to_end_numeric"] = num
                print(f"[orchestrator] end-to-end numeric: passed={num.get('passed')} "
                      f"{num.get('detail')}")
                e2e_ok = bool(num.get("passed"))

                # ------------- [4b] e2e_repair: feed the e2e failure back to ---
                #                    a fresh KernelAgent round and reinstall.
                # Without this loop, KernelAgent has no visibility into bugs that
                # LayerOracle passes but NetOracle catches (multi-input wiring,
                # post-install Net behavior, pnnx-specific input policy, ...).
                # Capped at e2e_repair_max_attempts to prevent runaway cost.
                adapter = None
                if self.adapt_e2e:
                    adapter = AdapterAgent(
                        task_name=self.task_name,
                        target_layer=(force_analog or kprof.get("analog_layer") or ""),
                        class_name=cls,
                        ncnn_root=self.ncnn_root,
                        llm_query=self.llm_query,
                        model=self.model,
                        run_dir=self.run_dir.parent / "adapter",
                    )
                attempt = 0
                while (not e2e_ok) and attempt < self.e2e_repair_max_attempts:
                    attempt += 1
                    print(f"\n===== [4b] e2e_repair (attempt {attempt}/"
                          f"{self.e2e_repair_max_attempts}, "
                          f"engine={'adapter' if adapter else 'kernel'}) =====")
                    print(f"[orchestrator] e2e detail: {num.get('detail')}")

                    if adapter is not None:
                        # contract-driven repair: AdapterAgent rewrites the
                        # already-correct algorithm to satisfy the ncnn
                        # Layer-Net contract, armed with the real .ncnn.param,
                        # the built-in reference impl, and the contract spec.
                        info = self._introspect_lite()
                        new_kcode = adapter.adapt(
                            kcode,
                            ncnn_param_text=self._e2e_param_text(graph_sum),
                            e2e_detail=str(num.get("detail")),
                            input_shapes=info.get("input_shapes"),
                            expected_out_shape=info.get("ncnn_output_shape"),
                            attempt=attempt,
                        )
                        if not new_kcode or new_kcode == kcode:
                            print(f"[orchestrator] adapter made no change on attempt "
                                  f"{attempt}; aborting repair loop")
                            break
                        summary["phases"][f"adapter_e2e_repair_{attempt}"] = {
                            "engine": "adapter",
                            "files": sorted(new_kcode.keys()),
                        }
                    else:
                        # legacy guess-driven repair (kept as a fallback path).
                        seed_fb = (f"end-to-end shape/value mismatch: {num.get('detail')}\n"
                                   f"Notes: the kernel's per-op LayerOracle is GREEN — the bug "
                                   f"is in how the kernel behaves when wired into the full "
                                   f"ncnn::Net (post-install).")
                        repair_sum = KernelAgent(
                            task_name=self.task_name, model_py=self.model_py,
                            cfg=self._cfg(run_numeric=True), llm_query=self.llm_query,
                            backend="base",
                            seed_code=kcode, seed_profile=kprof,
                            seed_feedback=seed_fb,
                            run_subdir_suffix=f"_e2e_repair_{attempt}",
                            force_analog_layer=force_analog,
                        ).run()
                        if repair_sum.get("status") != "success":
                            print(f"[orchestrator] e2e_repair attempt {attempt} did not "
                                  f"converge on LayerOracle; aborting repair loop")
                            break
                        new_kcode = (repair_sum.get("final_result") or {}).get("response_code") or {}
                        if not new_kcode:
                            break
                        summary["phases"][f"kernel_e2e_repair_{attempt}"] = {
                            "engine": "kernel",
                            "status": repair_sum.get("status"),
                            "rounds": repair_sum.get("rounds"),
                            "max_diff": (repair_sum.get("final_result") or {}).get("max_diff"),
                        }

                    # reinstall the repaired base layer, keep arm install as-is
                    if handle is not None:
                        netoc.restore(handle)
                    handle = netoc.install_layer(new_kcode, cls)
                    ok_rb, _ = netoc.rebuild_libncnn()
                    kcode = new_kcode
                    if not ok_rb:
                        print(f"[orchestrator] rebuild failed after repair attempt "
                              f"{attempt}; the repaired layer does not compile")
                        summary["phases"].setdefault(
                            f"adapter_e2e_repair_{attempt}" if adapter
                            else f"kernel_e2e_repair_{attempt}", {})["compiled"] = False
                        num = {"passed": False, "detail": "repaired layer failed to compile"}
                        summary["phases"]["end_to_end_numeric"] = num
                        continue
                    num = self._net_numeric(netoc, graph_sum, op_class=cls)
                    summary["phases"]["end_to_end_numeric"] = num
                    print(f"[orchestrator] e2e after repair attempt {attempt}: "
                          f"passed={num.get('passed')} {num.get('detail')}")
                    e2e_ok = bool(num.get("passed"))

            # ------------- [5] Production validation -------------
            if self.end_to_end and (e2e_ok is not False):
                prod_ok = self._production_validation(graph_sum, summary, op_class=cls)

            functional_ok = (
                kernel_ok and graph_ok
                and (e2e_ok if self.end_to_end else True)
                and (prod_ok if prod_ok is not None else True)
            )
            all_ok = functional_ok

            # ------------- [6] Optimization (real; opt-in) -------------
            if self.optimize and functional_ok:
                tgt = "arm" if (self.want_arm and arm_code) else "base"
                print(f"\n===== [6] OptimizeAgent (backend={tgt}, policy={self.optimize_policy}, "
                      f"map_budget={self.optimize_map_budget}) =====")
                opt_result, handle, arm_handle, kcode, arm_code = self._run_optimization(
                    kcode, kprof, graph_sum, summary, netoc, handle, arm_code, arm_handle)
                if opt_result.get("best_round", -1) >= 0 and opt_result.get("production_optimized_ok"):
                    print(f"[orchestrator] optimization improved + revalidated ({tgt}); "
                          f"new best perf={opt_result.get('best_perf', {}).get('avg')}")
                else:
                    print(f"[orchestrator] optimization kept baseline "
                          f"(stopped: {opt_result.get('stopped_reason')})")
        finally:
            if self.install and all_ok:
                print("\n===== [7] REGISTER: inject operator into ncnn/pnnx (permanent) =====")
                # if graph was skipped (already in ncnn), no pass to install
                if not already_in_ncnn:
                    summary["phases"]["register"] = self._register_pass(graph_sum, kcode, cls)
                else:
                    summary["phases"]["register"] = {
                        "registered": True, "ncnn_layer": cls,
                        "kernel_files": [f"src/layer/{n}" for n in kcode]
                                        + [f"src/layer/arm/{n}" for n in arm_code],
                        "pass_files": [], "pnnx_rebuilt": False,
                        "note": "ncnn already supports this op; only kernel was installed",
                    }
            elif self.end_to_end and handle is not None:
                print("\n===== [7] CLEANUP: restore ncnn source + rebuild libncnn clean =====")
                netoc.restore(handle)
                if arm_handle is not None: netoc.restore(arm_handle)
                netoc.rebuild_libncnn()

            # Belt-and-suspenders: even if any of the above teardown branches missed
            # something (or crashed), the guard's atexit/SIGTERM handler will catch
            # it. Calling it explicitly here surfaces any leftover IMMEDIATELY, in
            # the orchestrator log instead of at interpreter exit.
            if self._ncnn_guard is not None:
                try: self._ncnn_guard.restore()
                except Exception as exc:  # noqa: BLE001
                    print(f"[orchestrator] ncnn-guard restore failed: {exc}")

        summary["status"] = "success" if all_ok else summary.get("status", "fail")
        if self.install and all_ok:
            summary["registered"] = True
            summary["note"] = (
                "operator REGISTERED into ncnn/pnnx: kernel in src/layer + ncnn_add_layer"
                + (", pnnx pass installed" if not already_in_ncnn else " (graph already supported)")
                + "; native pnnx now converts it and ncnn runs it."
            )
        else:
            summary["note"] = (
                "kernel + (graph or skip) + end-to-end + production"
                + (" + optimization" if self.optimize else "")
                + " (temporary; source tree restored)."
            )
        write_json(self.run_dir / "summary.json", summary)
        print(f"\n[orchestrator] DONE status={summary['status']} "
              f"registered={summary.get('registered', False)}")
        return summary

    # --------------------------------------------------------- baseline probe
    def _early_baseline_probe(self) -> tuple[str | None, dict | None]:
        """Run pnnx baseline ONCE before KernelAgent, cache the result.

        Returns (force_analog_layer, grounding):
          - force_analog_layer: the principal ncnn layer type pnnx emits for
            this op (e.g. "Convolution", "Gemm", "AbsVal"), or None if pnnx
            failed / no convertible layer was found. Becomes a hard constraint
            on KernelAgent.analyze's analog_layer choice.
        The later existence-check reuses the cached grounding (no duplicate
        pnnx invocation).
        """
        rd = self.run_dir / "_baseline_probe"
        cfg = self._cfg(run_numeric=False)
        try:
            grounding = probe_pnnx_ir(cfg, self._resolve_model_py(), rd, self.task_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[orchestrator] baseline-probe failed: {exc}")
            self._cached_baseline = None
            return None, None
        self._cached_baseline = grounding
        ncnn_param_text = grounding.get("ncnn_param") or ""
        if not ncnn_param_text:
            return None, grounding
        from graph_pipeline import _ncnn_layer_types
        # Exclude pnnx-only emitted names (torch.logical_and, F.max_pool2d, ...):
        # they contain a namespace dot and are NOT real ncnn layers, so they must
        # not become force_analog (that would pin KernelAgent to a non-existent
        # analog and mis-drive the retarget guard). Such an op is unsupported ->
        # force_analog stays None -> GraphAgent authors the real conversion.
        op_layers = {t for t in _ncnn_layer_types(ncnn_param_text)
                     if t not in {"Input", "Output", "Split"} and "." not in t}
        if len(op_layers) == 1:
            return next(iter(op_layers)), grounding
        return None, grounding

    # --------------------------------------------------------- existence check
    def _check_already_in_ncnn(self) -> tuple[bool, dict]:
        """If the pnnx baseline already converts cleanly, mimic a graph_sum.

        Reuses the probe result cached by _early_baseline_probe (no duplicate
        pnnx run). Returns (already, graph_sum_like).
        """
        grounding = getattr(self, "_cached_baseline", None)
        if grounding is None:
            # _early_baseline_probe wasn't called or failed; fall back to
            # running pnnx ourselves (old behavior).
            rd = self.run_dir / "_baseline_probe"
            cfg = self._cfg(run_numeric=False)
            try:
                grounding = probe_pnnx_ir(cfg, self._resolve_model_py(), rd, self.task_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] existence probe failed: {exc}")
                return False, {}
        rd = self.run_dir / "_baseline_probe"
        supported = bool(grounding.get("baseline_supported"))
        op_types = grounding.get("op_types") or []
        residual = grounding.get("residual_aten") or []
        if not supported:
            return False, {"_baseline_op_types": op_types, "_residual_aten": residual}

        # build a graph_sum-like dict with the baseline-converted artifacts
        # Must include `_ncnn.py` — it's the pnnx-emitted per-blob squeeze policy
        # for ex.input() that NetOracle uses to feed inputs correctly. Without it
        # the squeeze policy degrades to torch_to_ncnn_input's blanket "drop axis 0",
        # which corrupts non-batched inputs like Conv weight tensors (16,3,3,3)
        # → (3,3,3) and silently produces wrong-channel outputs at e2e time.
        artifacts = {}
        for stem_suffix in (".pnnx.param", ".ncnn.param", ".ncnn.bin", "_ncnn.py"):
            for p in rd.rglob(f"*{stem_suffix}"):
                artifacts[stem_suffix] = str(p); break
        return True, {
            "status": "success",
            "rounds": 0,
            "final_result": {"artifacts": artifacts, "response_code": {}},
            "_baseline_op_types": op_types,
            "_residual_aten": residual,
        }

    # --------------------------------------------------------- permanent register
    def _register_pass(self, graph_sum: dict, kcode: dict, cls: str) -> dict:
        """Install the verified pnnx conversion pass permanently + rebuild pnnx."""
        from graph_pipeline import build_pnnx, inject_files
        from graph_schemas import BackupHandle

        gcode = (graph_sum.get("final_result") or {}).get("response_code") or {}
        pass_code = {k: v for k, v in gcode.items()
                     if k.split("/")[0] in ("pass_ncnn", "pass_level1", "pass_level2")}
        cfg = self._cfg(run_numeric=False)
        ok, _, err = inject_files(cfg, pass_code, BackupHandle())
        bok, _ = build_pnnx(cfg, self.run_dir / "register_pnnx_build.log")
        manifest = {
            "registered": ok and bok, "ncnn_layer": cls,
            "kernel_files": [f"src/layer/{n}" for n in kcode],
            "pass_files": [f"tools/pnnx/src/{k}" for k in pass_code],
            "pnnx_rebuilt": bok,
        }
        write_json(self.run_dir / "register_manifest.json", manifest)
        print(f"[orchestrator] registered: kernel={list(kcode)} pass={list(pass_code)} "
              f"pnnx_rebuilt={bok}")
        return manifest

    # --------------------------------------------------------- production validation
    def _production_validation(self, graph_sum: dict, summary: dict,
                               op_class: str | None = None) -> bool:
        """MoKA-style production validation appended after end-to-end numeric.

        Returns True if mandatory steps pass (benchmark never blocks success).
        """
        print("\n===== [5] PRODUCTION: compile + correctness"
              + (" + benchmark" if self.do_benchmark else "") + " =====")
        prod = self._run_production_step(self.model_py or self._resolve_model_py(), graph_sum,
                                         op_class=op_class)
        summary["phases"]["production"] = prod
        return bool(prod.get("_mandatory_ok"))

    def _run_production_step(self, model_py: str, graph_sum: dict,
                            op_class: str | None = None) -> dict:
        """Reusable production step. Returns {compile, correctness, profile?,
        _mandatory_ok}. Suitable as part of the optimization evaluator too.

        `op_class` (= Cand_<Op>) re-points the profiled model's output layer to
        OUR kernel so simpleperf measures it, not ncnn's built-in (needed when the
        op already existed in ncnn; idempotent otherwise).
        """
        from production_validation import ProductionValidator, torch_input_shapes_str
        pv = ProductionValidator(ncnn_root=self.ncnn_root, compile_mode=self.compile_mode,
                                 do_benchmark=self.do_benchmark, workdir=self.run_dir)
        pc = pv.production_compile()
        print(f"[orchestrator] production compile ({pc.get('mode')}): ok={pc.get('ok')}")
        cr = pv.production_correctness(graph_sum, model_py, retarget_to=op_class,
                                       expected_src_type=getattr(self, "_force_analog", None))
        print(f"[orchestrator] production correctness: passed={cr.get('passed')} "
              f"{cr.get('detail','')}")
        prod = {"compile": pc, "correctness": cr,
                "_mandatory_ok": bool(pc.get("ok") and cr.get("passed"))}
        if self.do_benchmark and prod["_mandatory_ok"]:
            art = (graph_sum.get("final_result") or {}).get("artifacts") or {}
            param = art.get(".ncnn.param")
            shapes = torch_input_shapes_str(model_py)
            # Single on-device path: profile_op runs benchncnn UNDER simpleperf, so
            # each per-thread config carries BOTH micro-arch metrics (IPC / cache-
            # miss / branch-miss / operator fraction) AND latency (min/max/avg).
            # No separate benchmark() run — the optimizer's baseline comes from here.
            prof = pv.profile_op(param, shapes, op_name=(op_class or self.task_name),
                                 retarget_to=op_class,
                                 expected_src_type=getattr(self, "_force_analog", None))
            if prof.get("ran"):
                c0 = (prof.get("configs") or [{}])[0]
                print(f"[orchestrator] op profile: ran=True ipc={c0.get('ipc')} "
                      f"cache_miss={c0.get('cache_miss_rate')} frac={c0.get('operator_fraction')} "
                      f"latency_avg={c0.get('latency_avg')} trust={c0.get('trustworthy')}")
            else:
                print(f"[orchestrator] op profile: skipped reason={prof.get('reason','')}")
            prod["profile"] = prof
        return prod

    # --------------------------------------------------------- optimization
    def _run_optimization(self, base_code: dict, kprof: dict, graph_sum: dict,
                          summary: dict, netoc: "NetOracle", handle,
                          arm_code: dict | None = None, arm_handle=None):
        """[6] Drive the REAL OptimizeAgent on the authored kernel.

        Backend-aware: if an arm kernel was authored, optimize THAT (compiling the
        base .cpp in as a fixed extra source, NC4HW4 packing); otherwise optimize
        the base kernel. The optimizer's LayerOracle clashes (duplicate symbol)
        with the kernel already in libncnn from the bridge, so we first restore a
        clean tree, optimize, then re-install base [+ arm winner] and re-validate
        the winner through production.

        Returns (opt_dict, handle, arm_handle, base_code, arm_code) with the chosen
        (winner-or-baseline) code installed and the handles refreshed for [7].
        """
        cls = kprof.get("class_name", "")
        model_py = self.model_py or self._resolve_model_py()
        weight_keys = list(kprof.get("weight_keys", []) or [])
        params = {int(k): v for k, v in (kprof.get("params") or {}).items()}
        arm_code = arm_code or {}
        target_arm = bool(self.want_arm and arm_code)   # optimize arm when available
        backend = "arm" if target_arm else "base"
        baseline_kernel = arm_code if target_arm else base_code

        # 1) clean tree so the optimizer's LayerOracle isn't a duplicate symbol
        for h in (handle, arm_handle):
            if h is not None: netoc.restore(h)
        netoc.rebuild_libncnn()

        # optimizer baseline latency: taken from the profile phase (profile_op runs
        # benchncnn under simpleperf; threads=1 is the cleanest single-op latency).
        # No separate benchmark() run — the same on-device measurement feeds both.
        prof = (summary.get("phases", {}).get("production", {}) or {}).get("profile", {}) or {}
        baseline_perf = _perf_from_profile(prof)

        from llm_api import query_llm
        llm = self.llm_query or query_llm
        agent = OptimizeAgent(
            task_name=self.task_name, baseline_kernel_code=baseline_kernel,
            model_py=model_py, ncnn_root=self.ncnn_root, llm_query=llm,
            model=self.model, weight_keys=weight_keys, params=params,
            baseline_perf=baseline_perf, policy=self.optimize_policy,
            max_rounds=self.max_optimize_rounds, improve_tol=self.improve_tol,
            inner_budget=self.optimize_inner_budget, map_budget=self.optimize_map_budget,
            coverage_target=self.optimize_coverage_target,
            experience_pool_path=self.experience_pool_path, op_class=cls,
            backend=backend, base_files=(base_code if target_arm else {}),
        )
        opt = agent.run().to_dict()
        opt["backend"] = backend
        summary["phases"]["optimization"] = opt
        write_json(self.run_dir / "optimization.json", opt)

        # 2) decide the winner; (re)install base + (arm winner-or-baseline)
        improved = opt.get("best_round", -1) >= 0 and opt.get("best_kernel")
        win_arm = opt["best_kernel"] if (improved and target_arm) else arm_code
        win_base = opt["best_kernel"] if (improved and not target_arm) else base_code

        new_handle = netoc.install_layer(win_base, cls)
        new_arm_handle = netoc.install_layer(win_arm, cls, subdir="arm", add_cmake=False) if win_arm else None
        ok, _ = netoc.rebuild_libncnn()
        if improved:
            prod = self._run_production_step(model_py, graph_sum, op_class=cls) if ok else {"_mandatory_ok": False}
            summary["phases"]["production_optimized"] = prod
            if ok and prod.get("_mandatory_ok"):
                opt["production_optimized_ok"] = True
            else:  # winner failed re-validation -> revert to baseline kernels
                opt["production_optimized_ok"] = False
                opt["note"] = "winner failed production re-validation; reverted to baseline"
                for h in (new_handle, new_arm_handle):
                    if h is not None: netoc.restore(h)
                new_handle = netoc.install_layer(base_code, cls)
                new_arm_handle = netoc.install_layer(arm_code, cls, subdir="arm", add_cmake=False) if arm_code else None
                netoc.rebuild_libncnn()
                win_base, win_arm = base_code, arm_code
        write_json(self.run_dir / "optimization.json", opt)
        return opt, new_handle, new_arm_handle, win_base, win_arm

    # ----------------------------------------------------------- net numeric
    def _net_numeric(self, netoc: NetOracle, graph_sum: dict,
                     op_class: str | None = None) -> dict:
        try:
            return self._net_numeric_impl(netoc, graph_sum, op_class=op_class)
        except Exception as exc:  # noqa: BLE001
            import traceback
            (self.run_dir / "net_numeric.log").write_text(
                traceback.format_exc(), encoding="utf-8")
            return {"passed": False,
                    "detail": f"net numeric raised: {type(exc).__name__}: {exc}"}

    def _e2e_param_text(self, graph_sum: dict) -> str:
        """The exact .ncnn.param the layer is fed at e2e time. Prefer the
        retargeted param (output layer re-pointed to our class) written by the
        last _net_numeric run; fall back to the graph's raw .ncnn.param."""
        rp = self.run_dir / "net_numeric_retargeted.param"
        if rp.exists():
            try:
                return rp.read_text(encoding="utf-8")
            except OSError:
                pass
        art = (graph_sum.get("final_result") or {}).get("artifacts") or {}
        p = art.get(".ncnn.param")
        if p and Path(p).exists():
            try:
                return Path(p).read_text(encoding="utf-8")
            except OSError:
                pass
        return ""

    def _introspect_lite(self) -> dict:
        """Cached input shapes + expected output shape for adapter prompts."""
        if getattr(self, "_cached_introspect", None) is not None:
            return self._cached_introspect
        info: dict = {}
        try:
            from kernel_pipeline import introspect_model
            info = introspect_model(self._resolve_model_py())
        except Exception:  # noqa: BLE001
            info = {}
        self._cached_introspect = info
        return info

    def _net_numeric_impl(self, netoc: NetOracle, graph_sum: dict,
                          op_class: str | None = None) -> dict:
        import torch
        from layer_oracle import retarget_param_output_file
        art = (graph_sum.get("final_result") or {}).get("artifacts") or {}
        param = art.get(".ncnn.param"); binf = art.get(".ncnn.bin")
        if not param or not binf or not Path(param).exists():
            return {"passed": False,
                    "detail": "no converted .ncnn.param/.bin from graph phase"}

        # re-point the output layer to OUR impl so the net runs ours, not the
        # built-in (needed for ops ncnn already supports; idempotent for new ops).
        if op_class:
            rp = self.run_dir / "net_numeric_retargeted.param"
            n = retarget_param_output_file(param, rp, op_class,
                                           expected_src_type=getattr(self, "_force_analog", None))
            param = str(rp)
            if n == 0:
                _fa = getattr(self, "_force_analog", None)
                why = (f"output layer type != principal `{_fa}`" if _fa
                       else "no single principal ncnn layer detected")
                print(f"[orchestrator] retarget SKIPPED ({why}) — op is a native "
                      f"multi-layer decomposition; running baseline graph as-is")

        import importlib.util
        mp = self._resolve_model_py()
        spec = importlib.util.spec_from_file_location("ds_model", mp)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        # Same fixed seed as the exported bin (graph_pipeline make_pt). Without
        # this, a weight-bearing op's reference model has different random weights
        # than the .ncnn.bin and e2e fails even when the kernel is correct.
        torch.manual_seed(0)
        model = (mod.Model(*init) if init else mod.Model()).eval()
        inputs = mod.get_inputs()
        with torch.no_grad():
            ref = model(*inputs)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        ref_np = ref.detach().numpy()
        in_names, out_name = parse_ncnn_io(Path(param).read_text(encoding="utf-8"))
        if len(in_names) != len(inputs):
            in_names = [f"in{i}" for i in range(len(inputs))]
        # pnnx-driven per-blob squeeze policy (falls back to drop-axis-0 when
        # _ncnn.py is missing or a blob name has no recorded policy).
        ncnn_py = art.get("_ncnn.py")
        ncnn_inputs = pnnx_driven_ncnn_inputs(inputs[:len(in_names)], in_names, ncnn_py)
        # Reference shape MUST match what ncnn produces. Mirror the squeeze policy
        # pnnx applied to in0: if axis 0 was squeezed (typical nn.Module ops where
        # axis 0 is the batch dim), drop it from the reference too. Otherwise
        # (e.g. ncnn Gemm: axis 0 is M, not batch — pnnx writes the .ncnn.py
        # with no squeeze), keep the full ref shape so the comparison aligns
        # with ncnn's output.
        in0_squeezed = (inputs[0].ndim >= 2
                        and ncnn_inputs[0].ndim == inputs[0].ndim - 1)
        if in0_squeezed and ref_np.ndim >= 2:
            reference = ref_np[0]
        else:
            reference = ref_np
        feed = {n: x for n, x in zip(in_names, ncnn_inputs)}
        out, log = netoc.run_net(param, binf, feed, out_name)
        (self.run_dir / "net_numeric.log").write_text(log, encoding="utf-8")
        if out is None:
            return {"passed": False, "detail": "net runner failed (see net_numeric.log)"}
        try:
            out_r = out.reshape(reference.shape)
        except ValueError:
            return {"passed": False,
                    "detail": f"shape mismatch ncnn {out.shape} vs ref {reference.shape}"}
        diff = np.abs(out_r - np.asarray(reference, dtype=np.float32))
        passed = bool(np.allclose(out_r, reference, atol=2e-3, rtol=2e-3))
        return {"passed": passed,
                "max_diff": float(diff.max()), "mean_diff": float(diff.mean()),
                "detail": f"max_diff={float(diff.max()):.6f} out_name={out_name} "
                          f"in_names={in_names}"}


def _perf_from_profile(prof: dict) -> dict:
    """Map a profile_op() result to the optimizer's baseline_perf shape ({avg,min}).

    profile_op runs benchncnn under simpleperf and attaches per-thread
    latency_{avg,min,max} to each config. We use threads=1 (cleanest single-op
    latency) when available, else the first config. Empty dict if no latency
    (e.g. profiling was skipped) — the optimizer then has no baseline to beat.
    """
    configs = (prof or {}).get("configs") or []
    if not configs:
        return {}
    chosen = next((c for c in configs if c.get("threads") == 1), configs[0])
    avg = chosen.get("latency_avg")
    if avg is None:
        return {}
    return {"avg": avg, "min": chosen.get("latency_min", avg),
            "threads": chosen.get("threads")}


def kprof_class_name(candidate: dict[str, str], baseline: dict[str, str]) -> str:
    """Best-effort: extract `class Xxx : public Layer` from candidate or baseline.

    Used by the optimization evaluator's evaluator() to know which class name
    to register. With the placeholder no-op proposer, candidate == baseline.
    """
    import re
    for src in (candidate, baseline):
        for _name, code in (src or {}).items():
            m = re.search(r"class\s+(\w+)\s*:\s*public\s+Layer", code or "")
            if m:
                return m.group(1)
    return ""
