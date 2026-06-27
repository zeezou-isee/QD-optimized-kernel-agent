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


def _resolve_model_py(task: str, explicit: str | None, dataset_root: str | None) -> str:
    if explicit:
        return str(explicit)
    base = Path(dataset_root) if dataset_root else (
        KERNELGEN_ROOT / "MobileKernelBench_git" / "dataset" / "Mobilekernelbench")
    hits = sorted(Path(base).rglob(f"{task}.py"))
    if not hits:
        raise FileNotFoundError(f"{task}.py not found under {base}")
    return str(hits[0])


def _kernel_from_summary(task: str, sub: str) -> dict[str, str]:
    summ = RUNS_ROOT / task / sub / "summary.json"
    if summ.exists():
        data = json.loads(summ.read_text(encoding="utf-8"))
        return (data.get("final_result") or {}).get("response_code") or {}
    return {}


def _load_baseline_kernel(task: str, kernel_dir: str | None, backend: str) -> dict[str, str]:
    """Read the kernel to optimize. base -> runs/<task>/kernel; arm -> kernel_arm;
    vulkan -> kernel_vulkan (a .h/.cpp + .comp triple)."""
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
    sub = "kernel" if backend == "base" else f"kernel_{backend}"
    code = _kernel_from_summary(task, sub)
    if code:
        return code
    raise FileNotFoundError(
        f"no {backend} baseline kernel (run `run_kernel_agent --task {task} "
        f"--backend {backend}` first, or pass --kernel-dir)")


def main() -> None:
    p = argparse.ArgumentParser(description="M1 kernel optimizer (CPU, LLM proposer + inner search).")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None, help="PyTorch reference .py (else dataset lookup)")
    p.add_argument("--model-name", default="z-ai/glm-5.1", help="OpenRouter model for the proposer")
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
    p.add_argument("--backend", choices=["base", "arm", "vulkan"], default="base",
                   help="base = portable C++; arm = NEON/NC4HW4 kernel; vulkan = GPU kernel "
                        "(.cpp + .comp, optimized & measured on a Vulkan device). "
                        "arm/vulkan need a verified base + that-backend kernel from run_kernel_agent")
    args = p.parse_args()

    model_py = _resolve_model_py(args.task, args.model, args.dataset_root)
    baseline = _load_baseline_kernel(args.task, args.kernel_dir, args.backend)
    ncnn_root = args.ncnn_root or GraphConfig().ncnn_root
    # arm/vulkan subclass the verified base -> compile the base .cpp in as a fixed source
    base_files = _kernel_from_summary(args.task, "kernel") if args.backend in ("arm", "vulkan") else {}

    agent = OptimizeAgent(
        task_name=args.task, baseline_kernel_code=baseline,
        model_py=model_py, ncnn_root=ncnn_root, llm_query=query_llm,
        model=args.model_name, max_rounds=args.max_rounds,
        inner_budget=args.inner_budget, runs=args.runs, warmup=args.warmup,
        improve_tol=args.improve_tol,
        policy=args.policy, map_budget=args.map_budget,
        coverage_target=args.coverage_target, patience=args.patience,
        regime=args.regime, experience_pool_path=args.experience_pool,
        run_baseline_comparison=args.baseline_compare, op_class=args.task,
        backend=args.backend, base_files=base_files,
    )
    res: OptimizeResult = agent.run()

    out_dir = RUNS_ROOT / args.task / "optimize"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
