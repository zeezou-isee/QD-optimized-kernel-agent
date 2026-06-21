"""CLI for KernelAgent — write an ncnn base kernel from scratch + verify vs PyTorch.

    python run_kernel_agent.py --task Abs --model-name z-ai/glm-5.1
    python run_kernel_agent.py --task ELU --max-rounds 8
    python run_kernel_agent.py --task PRelu --model /path/to/PRelu.py
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

from config import GraphConfig, RUNS_ROOT
from kernel_agent import KernelAgent


def _load_base_kernel(task: str, base_dir: str | None) -> tuple[dict, dict]:
    """For --backend arm: load verified base kernel code + profile from a prior
    base run (runs/<task>/kernel/summary.json) or an explicit --base-kernel-dir."""
    if base_dir:
        d = Path(base_dir)
        code = {p.name: p.read_text(encoding="utf-8") for p in d.glob("*")
                if p.suffix in (".h", ".hpp", ".cpp", ".cc", ".cxx")}
        prof = {}
        pj = d / "kernel_profile.json"
        if pj.exists():
            prof = json.loads(pj.read_text(encoding="utf-8"))
        return code, prof
    summ = RUNS_ROOT / task / "kernel" / "summary.json"
    if not summ.exists():
        raise FileNotFoundError(f"no base kernel found; run --backend base first or pass "
                                f"--base-kernel-dir (looked at {summ})")
    data = json.loads(summ.read_text(encoding="utf-8"))
    code = (data.get("final_result") or {}).get("response_code") or {}
    prof = data.get("kernel_profile") or {}
    return code, prof


def main() -> None:
    p = argparse.ArgumentParser(description="From-scratch ncnn kernel writer (base/arm) + PyTorch oracle.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None, help="Path to the PyTorch reference .py (overrides dataset lookup).")
    p.add_argument("--model-name", default="z-ai/glm-5.1", help="LLM model id (OpenRouter).")
    p.add_argument("--ncnn-root", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--no-numeric", action="store_true", help="compile-only (skip allclose)")
    p.add_argument("--backend", choices=["base", "arm"], default="base",
                   help="base = portable C++; arm = NEON/NC4HW4 subclass of the base layer")
    p.add_argument("--base-kernel-dir", default=None,
                   help="arm: dir with the verified base cand_*.{h,cpp} + kernel_profile.json "
                        "(default: runs/<task>/kernel)")
    args = p.parse_args()

    cfg = GraphConfig(
        ncnn_root=Path(args.ncnn_root) if args.ncnn_root else GraphConfig().ncnn_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        model=args.model_name,
        max_rounds=args.max_rounds,
        run_numeric=not args.no_numeric,
    )
    base_code, base_prof = ({}, {})
    if args.backend == "arm":
        base_code, base_prof = _load_base_kernel(args.task, args.base_kernel_dir)
    agent = KernelAgent(task_name=args.task, model_py=args.model, cfg=cfg,
                        backend=args.backend, base_kernel_code=base_code, base_profile=base_prof)
    summary = agent.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
