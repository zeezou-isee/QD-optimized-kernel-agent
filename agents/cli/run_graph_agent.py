"""CLI entry point for GraphAgent.

Examples:
    # use a dataset model by task name (auto-located under the dataset root)
    python run_graph_agent.py --task HardSigmoid

    # point at a specific PyTorch reference file
    python run_graph_agent.py --task MyOp --model /path/to/MyOp.py --max-rounds 8

    # structural-only (skip end-to-end allclose, no ncnn kernel required)
    python run_graph_agent.py --task MyOp --no-numeric

    # keep the injected pass files + CMake edits after success
    python run_graph_agent.py --task MyOp --keep-on-success
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# bootstrap agents flat-import paths (we live in agents/cli/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # need EndtoEnd... so `import agents` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agents/ for flat imports
import agents as _agents; _agents.bootstrap_paths()

from config import GraphConfig
from graph_agent import GraphAgent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate & verify an ncnn graph-conversion for a PyTorch operator.")
    p.add_argument("--task", required=True, help="Operator task name (and dataset file stem).")
    p.add_argument("--model", default=None, help="Path to the PyTorch reference .py (overrides dataset lookup).")
    p.add_argument("--model-code", default=None, help="Inline PyTorch model source (overrides --model).")
    p.add_argument("--ncnn-root", default=None, help="Path to the ncnn source tree.")
    p.add_argument("--dataset-root", default=None, help="Path to the PyTorch reference dataset root.")
    p.add_argument("--model-name", default="anthropic/claude-sonnet-4.5", help="LLM model id (OpenRouter).")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--build-jobs", type=int, default=8)
    p.add_argument("--torch-install-dir", default=None, help="Optional libtorch install dir for pnnx cmake.")
    p.add_argument("--no-numeric", action="store_true", help="Skip end-to-end allclose (structural only).")
    p.add_argument("--keep-on-success", action="store_true", help="Keep injected files/CMake edits after success.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = GraphConfig(
        ncnn_root=Path(args.ncnn_root) if args.ncnn_root else GraphConfig().ncnn_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        model=args.model_name,
        max_rounds=args.max_rounds,
        build_jobs=args.build_jobs,
        torch_install_dir=Path(args.torch_install_dir) if args.torch_install_dir else None,
        run_numeric=not args.no_numeric,
        keep_changes_on_success=args.keep_on_success,
    )
    agent = GraphAgent(
        task_name=args.task,
        model_py=args.model,
        model_code=args.model_code,
        cfg=cfg,
    )
    summary = agent.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
