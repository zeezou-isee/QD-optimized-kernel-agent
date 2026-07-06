"""CLI for the OperatorAgent orchestrator (coupled + end-to-end numeric).

    python run_operator_agent.py --task Greater --model-name z-ai/glm-5.1
    python run_operator_agent.py --task Greater --no-end-to-end   # kernel numeric + graph structural only
    python run_operator_agent.py --task Greater --keep-installed   # leave the kernel installed in ncnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# bootstrap opgen flat-import paths (we live in opgen/cli/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # need EndtoEnd... so `import opgen` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # opgen/ for flat imports
import opgen as _opgen; _opgen.bootstrap_paths()

from operator_agent import OperatorAgent


def main() -> None:
    p = argparse.ArgumentParser(description="From-scratch ncnn operator: kernel + conversion, coupled + e2e numeric.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--model-name", default="deepseek-v4-pro")
    p.add_argument("--max-rounds", type=int, default=15)
    p.add_argument("--ncnn-root", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--torch-install-dir", default=None)
    p.add_argument("--no-end-to-end", action="store_true",
                   help="skip install+rebuild+net-numeric (kernel numeric + graph structural only)")
    p.add_argument("--install", action="store_true",
                   help="PERMANENTLY register the verified operator into ncnn/pnnx "
                        "(kernel -> src/layer + ncnn_add_layer, pnnx pass installed, rebuilt). "
                        "Default off = temporary verify then restore.")
    p.add_argument("--compile-mode", choices=["build_lib", "build_full"], default="build_lib",
                   help="production compile mode. build_lib (default) reuses libncnn.a; "
                        "build_full does a MoKA-style full ncnn build with examples+tests.")
    p.add_argument("--benchmark", action="store_true",
                   help="run android+adb benchncnn benchmark after correctness. "
                        "Auto-skipped (NOT a failure) if ANDROID_NDK is unset or no device.")
    p.add_argument("--graph-max-rounds", type=int, default=15,
                   help="GraphAgent iteration cap (default 15). If graph fails within "
                        "this many rounds, the operator is aborted.")
    p.add_argument("--optimize", action="store_true",
                   help="after functional+production pass, drive the REAL OptimizeAgent "
                        "(LLM proposer + inner search / MAP-Elites) to improve perf; "
                        "winner is re-validated through production.")
    p.add_argument("--max-optimize-rounds", type=int, default=15,
                   help="OptimizeAgent rounds (linear policy)")
    p.add_argument("--improve-tol", type=float, default=0.02,
                   help="convergence threshold for optimization (<2%% improvement -> stop)")
    p.add_argument("--optimize-policy", choices=["linear", "map_elites"], default="map_elites",
                   help="linear (M1 single-template loop) | map_elites (M2/M3 QD outer loop)")
    p.add_argument("--optimize-map-budget", type=int, default=60,
                   help="map_elites total measurement budget")
    p.add_argument("--optimize-inner-budget", type=int, default=8,
                   help="per-template inner search measurement budget")
    p.add_argument("--optimize-coverage-target", type=int, default=4,
                   help="map_elites niches to fill before switching to quality bias")
    p.add_argument("--experience-pool", default=None,
                   help="path to the 兵器谱 JSON (warm-start seeds + persist on finish)")
    p.add_argument("--backends", default="base",
                   help="comma list of kernel backends to author/install: base[,arm]. "
                        "arm adds a NEON/NC4HW4 subclass; the conversion graph targets the "
                        "base class and ncnn auto-selects arm at runtime.")
    p.add_argument("--allow-backend-fallback", action="store_true",
                   help="if a requested target backend (e.g. arm) fails, degrade to base-only "
                        "instead of aborting. OFF by default: a target backend is a hard gate.")
    p.add_argument("--auto-cleanup", action="store_true",
                   help="ncnn-tree guard: if the ncnn source tree is found dirty at startup "
                        "(typically leaked from a prior killed run), silently clean it instead "
                        "of aborting. OFF by default — abort is safer; turn on for batch jobs.")
    p.add_argument("--device-verify", choices=["off", "auto", "on"], default="off",
                   help="device-in-the-loop gate: after each KernelAgent round's HOST verify passes, "
                        "also verify on the REAL phone (base/arm) and feed device failures back to "
                        "the LLM. auto = device if detected else host-only; on = warn if none; "
                        "off (default) = host-only.")
    p.add_argument("--device-simpleperf", action="store_true",
                   help="device gate also collects PMU via simpleperf (default off).")
    args = p.parse_args()

    summary = OperatorAgent(
        task_name=args.task, model_py=args.model, model=args.model_name,
        max_rounds=args.max_rounds, graph_max_rounds=args.graph_max_rounds,
        ncnn_root=args.ncnn_root, dataset_root=args.dataset_root,
        torch_install_dir=args.torch_install_dir,
        end_to_end=not args.no_end_to_end, install=args.install,
        compile_mode=args.compile_mode, benchmark=args.benchmark,
        optimize=args.optimize, max_optimize_rounds=args.max_optimize_rounds,
        improve_tol=args.improve_tol, optimize_policy=args.optimize_policy,
        optimize_map_budget=args.optimize_map_budget,
        optimize_inner_budget=args.optimize_inner_budget,
        optimize_coverage_target=args.optimize_coverage_target,
        experience_pool_path=args.experience_pool,
        backends=[b.strip() for b in args.backends.split(",") if b.strip()],
        allow_backend_fallback=args.allow_backend_fallback,
        auto_cleanup_ncnn=args.auto_cleanup,
        device_verify=args.device_verify,
        device_simpleperf=args.device_simpleperf,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
