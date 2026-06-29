"""Mobilekernelbench subset (~30 ops) — mid-tier coverage set for batch_runner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET          = ROOT / "dataset" / "Mobilekernelbench_subset"
MODEL            = "deepseek-v4-pro"
MAX_ROUNDS       = "15"
GRAPH_MAX_ROUNDS = "10"
PER_OP_TIMEOUT   = 1800            # 30 min hard cap per op
BACKENDS         = "base,arm"
COMPILE_MODE     = "build_lib"
