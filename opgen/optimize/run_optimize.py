"""Standalone CLI for the M1 OptimizeAgent (real inner loop).

    # optimize a kernel whose verified baseline already lives in runs/<task>/kernel
    python optimize/run_optimize.py --task Abs --model-name z-ai/glm-5.1

    # or point at a directory holding the baseline cand_*.h / cand_*.cpp
    python optimize/run_optimize.py --task Abs --kernel-dir runs/Abs/kernel/round_00

Resolves the PyTorch reference model from the dataset (or --model), loads the
verified baseline kernel, then runs Proposer -> inner_search -> best.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # EndtoEnd.. for `import opgen`
import opgen as _opgen; _opgen.bootstrap_paths()

from config import KERNELGEN_ROOT, RUNS_ROOT, GraphConfig
from llm_api import query_llm
from optimize_agent import OptimizeAgent
from schemas import OptimizeResult
import paths


def _resolve_model_py(task: str, explicit: str | None, dataset_root: str | None) -> str:
    if explicit:
        return str(explicit)
    base = Path(dataset_root) if dataset_root else (
        KERNELGEN_ROOT / "MobileKernelBench_git" / "dataset" / "Mobilekernelbench")
    hits = sorted(Path(base).rglob(f"{task}.py"))
    if not hits:
        raise FileNotFoundError(f"{task}.py not found under {base}")
    return str(hits[0])


def _kernel_from_summary(task: str, backend: str, runs_root: Path | None = None) -> dict[str, str]:
    """Load kernel `response_code` for (task, backend), preferring new layout.
    Routes through paths.kernel_summary which handles legacy fallback."""
    root = runs_root or RUNS_ROOT
    summ = paths.kernel_summary(root, task, backend)
    if summ.exists():
        data = json.loads(summ.read_text(encoding="utf-8"))
        return (data.get("final_result") or {}).get("response_code") or {}
    return {}


def _load_baseline_kernel(task: str, kernel_dir: str | None, backend: str,
                          runs_root: Path | None = None) -> dict[str, str]:
    """Read the kernel to optimize. NEW layout:
      base   -> runs/<task>/base_kernel/artifacts/  (SoT contract dir)
      arm    -> runs/<task>/backends/arm/kernel/summary.json
      vulkan -> runs/<task>/backends/vulkan/kernel/summary.json  (.h/.cpp/.comp)
    Legacy fallbacks handled inside paths.kernel_summary.
    """
    if kernel_dir:
        d = Path(kernel_dir)
        exts = (".h", ".hpp", ".cpp", ".cc", ".cxx") + ((".comp",) if backend == "vulkan" else ())

        def _keep(name: str) -> bool:
            if backend == "arm":
                return name.endswith(("_arm.h", "_arm.cpp"))
            if backend == "vulkan":
                return name.endswith(("_vulkan.h", "_vulkan.cpp", ".comp"))
            return not name.endswith(("_arm.h", "_arm.cpp", "_vulkan.h", "_vulkan.cpp"))

        code = {p.name: p.read_text(encoding="utf-8") for p in d.glob("*")
                if p.suffix in exts and _keep(p.name)}
        if code:
            return code
        raise FileNotFoundError(f"no matching kernel files for backend={backend} in {d}")

    root = runs_root or RUNS_ROOT
    # base: prefer artifacts/ contract dir before falling back to summary
    if backend == "base":
        art = paths.base_kernel_artifacts_dir(root, task)
        if art.exists():
            code = {p.name: p.read_text(encoding="utf-8") for p in art.glob("*")
                    if p.suffix in (".h", ".hpp", ".cpp", ".cc", ".cxx")}
            if code:
                return code
    code = _kernel_from_summary(task, backend, runs_root=root)
    if code:
        return code
    raise FileNotFoundError(
        f"no {backend} baseline kernel (run `run_kernel_agent --task {task} "
        f"--backend {backend}` first, or pass --kernel-dir)")


def main() -> None:
    p = argparse.ArgumentParser(description="M1 kernel optimizer (CPU, LLM proposer + inner search).")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None, help="PyTorch reference .py (else dataset lookup)")
    p.add_argument("--model-name", default="deepseek-v4-pro",
                   help="LLM model id for the proposer (DeepSeek by default; OpenRouter ids also work)")
    p.add_argument("--kernel-dir", default=None, help="dir with baseline cand_*.h/.cpp")
    p.add_argument("--ncnn-root", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--inner-budget", type=int, default=10)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--improve-tol", type=float, default=0.02)
    # M2/M3
    p.add_argument("--policy", choices=["linear", "map_elites"], default="linear",
                   help="linear (M1 single-template loop) | map_elites (M2/M3 QD outer loop)")
    p.add_argument("--map-budget", type=int, default=80, help="map_elites total measurement budget")
    p.add_argument("--coverage-target", type=int, default=4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--regime", choices=["memory_bound", "compute_bound"], default=None,
                   help="override roofline regime (else auto from arithmetic intensity)")
    p.add_argument("--experience-pool", default=None,
                   help="path to the 兵器谱 JSON (warm-start seeds + persist on finish)")
    p.add_argument("--baseline-compare", action="store_true",
                   help="also run a best-first control arm and report the verdict (§7.5)")
    p.add_argument("--n-promote", type=int, default=3,
                   help="axis-extension: a novel structural label must win/open a niche "
                        "in this many DISTINCT tasks before it is promoted into Σ "
                        "(experience_pool/wiki/sigma/<backend>.json). Default 3.")
    p.add_argument("--backend", choices=["base", "arm", "vulkan"], default="base",
                   help="base = portable C++; arm = NEON/NC4HW4 kernel; vulkan = GPU kernel "
                        "(.cpp + .comp, optimized & measured on a Vulkan device). "
                        "arm/vulkan need a verified base + that-backend kernel from run_kernel_agent")
    p.add_argument("--device-verify", choices=["off", "auto", "on"], default="off",
                   help="measure candidate + baseline latency on the REAL phone (auto = use "
                        "device if adb sees one, else host wall-clock). MANDATORY for a "
                        "meaningful latency objective — host subprocess wall-clock is not phone time.")
    p.add_argument("--bench-loop", type=int, default=100,
                   help="on-device timed forwards per measurement (default 100). Records min/max/avg; "
                        "the search objective + reported latency use avg.")
    p.add_argument("--bench-warmup", type=int, default=10,
                   help="on-device discarded warmup forwards before timing (default 10)")
    p.add_argument("--crossover-rate", type=float, default=0.4,
                   help="MAP-Elites P(crossover) per round (0 = mutation-only; default 0.4). "
                        "For the crossover ablation.")
    p.add_argument("--fill-budget", type=int, default=16,
                   help="two-phase illumination: Phase-1 batch-proposed seed templates, filled "
                        "at 1 eval each (thickens the grid cheaply). Default 16.")
    p.add_argument("--optimize-topk", type=int, default=6,
                   help="two-phase illumination: Phase-2 niches to deepen with full inner_search "
                        "(bounds the deep-search burden ≈ topk × inner_budget). Default 6.")
    p.add_argument("--record-trace", action="store_true",
                   help="persist the full inner-search trace (per-round climb trajectory, "
                        "analytically-pruned points + reasons, param space) + bd_axes/inner_config "
                        "into summary.json for paper visualization. Bloats the summary — use for "
                        "case-study ops. Export with scripts/optimize_trace.py.")
    p.add_argument("--runs-root", default=None,
                   help="override the root that holds runs/<task>/{kernel,kernel_arm,...} "
                        "when loading baselines. Default: opgen/runs")
    p.add_argument("--out-dir", default=None,
                   help="override where summary.json is written. Default: "
                        "<runs-root>/<task>/backends/<backend>/optimize/ (new 5-stage layout)")
    args = p.parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else RUNS_ROOT

    model_py = _resolve_model_py(args.task, args.model, args.dataset_root)
    baseline = _load_baseline_kernel(args.task, args.kernel_dir, args.backend,
                                     runs_root=runs_root)
    ncnn_root = args.ncnn_root or GraphConfig().ncnn_root
    # arm/vulkan subclass the verified base -> compile the base .cpp in as a
    # fixed source. Prefer the artifacts/ contract dir, fall back to summary
    # (both handled by _load_baseline_kernel for backend="base").
    base_files = _load_baseline_kernel(args.task, None, "base", runs_root=runs_root) \
        if args.backend in ("arm", "vulkan") else {}

    # weight_keys + params from the kernel profile — WITHOUT these the evaluator
    # passes NO weights to the baseline, so weighted ops (BatchNorm/Conv/Gemm)
    # crash at load_model ("load_model failed"). Prefer the analyze/ shared
    # profile, fall back to the base_kernel artifacts profile.
    weight_keys, params = [], {}
    # legacy runs_arm/runs_vulkan trees keep the profile at kernel/kernel_profile.json
    # or kernel_<backend>/kernel_profile.json — need those fallbacks too.
    _task_root = runs_root / args.task
    _profile_candidates = [
        paths.kernel_profile_shared_json(runs_root, args.task),
        paths.base_kernel_artifacts_dir(runs_root, args.task) / "kernel_profile.json",
        _task_root / "kernel" / "kernel_profile.json",
        _task_root / f"kernel_{args.backend}" / "kernel_profile.json",
    ]
    for pp in _profile_candidates:
        if pp.exists():
            prof = json.loads(pp.read_text(encoding="utf-8"))
            weight_keys = list(prof.get("weight_keys") or [])
            params = {int(k): v for k, v in (prof.get("params") or {}).items()}
            break

    # pnnx-emitted _ncnn.py holds the per-blob input squeeze policy; without it the
    # evaluator falls back to blanket "drop axis 0", which corrupts batch-less
    # matrices (MatMul/Einsum). Resolve it from the op's cached pnnx probe.
    _ncnn_py = next(iter(sorted((runs_root / args.task).rglob("*_ncnn.py"))), None)

    agent = OptimizeAgent(
        task_name=args.task, baseline_kernel_code=baseline,
        model_py=model_py, ncnn_root=ncnn_root, llm_query=query_llm,
        model=args.model_name, max_rounds=args.max_rounds,
        inner_budget=args.inner_budget, runs=args.runs, warmup=args.warmup,
        improve_tol=args.improve_tol, weight_keys=weight_keys, params=params,
        policy=args.policy, map_budget=args.map_budget,
        coverage_target=args.coverage_target, patience=args.patience,
        regime=args.regime, experience_pool_path=args.experience_pool,
        run_baseline_comparison=args.baseline_compare, op_class=args.task,
        backend=args.backend, base_files=base_files, n_promote=args.n_promote,
        device_measure=(args.device_verify in ("auto", "on")),
        ncnn_py=_ncnn_py, record_trace=args.record_trace,
        device_bench=args.bench_loop, device_warmup=args.bench_warmup,
        crossover_rate=args.crossover_rate,
        fill_budget=args.fill_budget, optimize_topk=args.optimize_topk,
    )
    res: OptimizeResult = agent.run()

    out_dir = (Path(args.out_dir) if args.out_dir
               else paths.backend_optimize_dir(runs_root, args.task, args.backend))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
