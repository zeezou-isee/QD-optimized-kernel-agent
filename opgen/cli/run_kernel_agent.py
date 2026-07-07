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
import paths


def _load_base_kernel(task: str, base_dir: str | None) -> tuple[dict, dict]:
    """For --backend arm/vulkan: load verified base kernel code + profile.

    Resolution order:
    1. Explicit --base-kernel-dir (user override).
    2. NEW layout: runs/<task>/base_kernel/artifacts/ (published as a CONTRACT
       by KernelAgent._publish_base_artifacts on successful base run).
    3. Legacy: runs/<task>/base_kernel/summary.json (in-place summary read).
    4. Pre-restructure legacy: runs/<task>/kernel/summary.json.
    """
    if base_dir:
        d = Path(base_dir)
        code = {p.name: p.read_text(encoding="utf-8") for p in d.glob("*")
                if p.suffix in (".h", ".hpp", ".cpp", ".cc", ".cxx")}
        prof = {}
        pj = d / "kernel_profile.json"
        if pj.exists():
            prof = json.loads(pj.read_text(encoding="utf-8"))
        return code, prof

    # 2. artifacts/ contract dir — the SoT for cross-backend consumption
    art = paths.base_kernel_artifacts_dir(RUNS_ROOT, task)
    if art.exists():
        code = {p.name: p.read_text(encoding="utf-8") for p in art.glob("*")
                if p.suffix in (".h", ".hpp", ".cpp", ".cc", ".cxx")}
        if code:
            prof = {}
            pj = art / "kernel_profile.json"
            if pj.exists():
                prof = json.loads(pj.read_text(encoding="utf-8"))
            return code, prof

    # 3+4. summary.json fallback (new base_kernel dir first, then legacy kernel/)
    summ = paths.kernel_summary(RUNS_ROOT, task, "base")
    if not summ.exists():
        raise FileNotFoundError(f"no base kernel found; run --backend base first or pass "
                                f"--base-kernel-dir (looked at {art} and {summ})")
    data = json.loads(summ.read_text(encoding="utf-8"))
    code = (data.get("final_result") or {}).get("response_code") or {}
    prof = data.get("kernel_profile") or {}
    return code, prof


def main() -> None:
    p = argparse.ArgumentParser(description="From-scratch ncnn kernel writer (base/arm) + PyTorch oracle.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None, help="Path to the PyTorch reference .py (overrides dataset lookup).")
    p.add_argument("--model-name", default="deepseek-v4-pro",
                   help="LLM model id (DeepSeek by default; OpenRouter ids like 'anthropic/...' also work).")
    p.add_argument("--ncnn-root", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--no-numeric", action="store_true", help="compile-only (skip allclose)")
    p.add_argument("--backend", choices=["base", "arm", "vulkan"], default="base",
                   help="base = portable C++; arm = NEON/NC4HW4 subclass; "
                        "vulkan = GPU subclass (.cpp + .comp, verified on a Vulkan device)")
    p.add_argument("--base-kernel-dir", default=None,
                   help="arm/vulkan: dir with the verified base cand_*.{h,cpp} + kernel_profile.json "
                        "(default: runs/<task>/kernel)")
    p.add_argument("--vulkan-mode", choices=["scratch", "native_first", "native_only"],
                   default="scratch",
                   help="--backend vulkan: dispatch mode. scratch (default) = agent authors "
                        ".h+.cpp+.comp shader from scratch; native_first = try native subclass, "
                        "fall back to scratch on non-verify; native_only = never asks the LLM to "
                        "write a shader (legacy miniset audit path).")
    p.add_argument("--device-verify", choices=["off", "auto", "on"], default="off",
                   help="device-in-the-loop gate: after each round's HOST verify passes, also "
                        "verify on the REAL phone (base/arm; correctness + latency) and feed device "
                        "failures back to the LLM. auto = use device if detected else host-only; "
                        "on = same but warn if no device; off (default) = host-only.")
    p.add_argument("--device-simpleperf", action="store_true",
                   help="device gate also collects PMU via simpleperf (default off: correctness + "
                        "plain latency only).")
    p.add_argument("--no-device-speedup", action="store_true",
                   help="disable the inline speedup measurement (by default the device gate also "
                        "times the native ncnn op via create_layer on the SAME runner -> fair "
                        "single-layer speedup, zero extra compile).")
    args = p.parse_args()

    cfg = GraphConfig(
        ncnn_root=Path(args.ncnn_root) if args.ncnn_root else GraphConfig().ncnn_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        model=args.model_name,
        max_rounds=args.max_rounds,
        run_numeric=not args.no_numeric,
        vulkan_mode=args.vulkan_mode,
    )
    base_code, base_prof = ({}, {})
    if args.backend in ("arm", "vulkan"):
        base_code, base_prof = _load_base_kernel(args.task, args.base_kernel_dir)
    agent = KernelAgent(task_name=args.task, model_py=args.model, cfg=cfg,
                        backend=args.backend, base_kernel_code=base_code, base_profile=base_prof,
                        device_verify=args.device_verify, device_simpleperf=args.device_simpleperf,
                        device_speedup=not args.no_device_speedup)
    summary = agent.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
