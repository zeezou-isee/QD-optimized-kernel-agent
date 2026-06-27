"""agents — ncnn operator generation agent package.

The codebase uses flat (top-level) imports throughout — e.g. inside graph/graph_agent.py
you'll see `from graph_pipeline import ...`, not `from agents.graph.graph_pipeline import ...`.
This keeps the diffs small across the reorg and lets every module be re-runnable on its own.

To make those flat imports resolve regardless of which subdirectory a module lives in,
all entry points must call `bootstrap_paths()` once before importing anything from agents.

Usage from a CLI script:
    import agents; agents.bootstrap_paths()
    from operator_agent import OperatorAgent  # works because bootstrap_paths put it on sys.path

The function is idempotent and side-effect-only (it just prepends each agents subdir to sys.path).
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parent

# Subdirectories that contain flat-importable Python modules.
_FLAT_SUBDIRS = (
    "",                # agents itself (config.py, llm_api.py)
    "graph",
    "kernel",
    "orchestrator",
    "optimize",
    "layer_oracle",    # already a self-contained package, but its package dir
                       # must also be on sys.path so `from layer_oracle import ...` works
    "tools",
)

_BOOTSTRAPPED = False


def bootstrap_paths() -> None:
    """Prepend each agents subdirectory to sys.path so flat imports resolve.

    Idempotent — calling more than once is a no-op.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for sub in _FLAT_SUBDIRS:
        p = AGENTS_ROOT / sub if sub else AGENTS_ROOT
        if p.is_dir():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
    _BOOTSTRAPPED = True


# Also bootstrap on import (B in the plan): so `python -m agents.cli.run_operator_agent` or
# `from agents.orchestrator.operator_agent import OperatorAgent` also work without extra setup.
bootstrap_paths()

__all__ = ["bootstrap_paths", "AGENTS_ROOT"]
