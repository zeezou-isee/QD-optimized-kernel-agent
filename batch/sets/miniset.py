"""Mobilekernelbench miniset (~9 ops) — fast smoke set for batch_runner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET          = ROOT / "dataset" / "Mobilekernelbench_miniset"
MODEL            = "deepseek-v4-pro"
MAX_ROUNDS       = "15"
GRAPH_MAX_ROUNDS = "10"
PER_OP_TIMEOUT   = 3600            # 60 min hard cap per op (Conv/Gemm need it)
BACKENDS         = "base,arm"
COMPILE_MODE     = "build_lib"
