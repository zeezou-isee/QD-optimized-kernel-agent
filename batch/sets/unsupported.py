"""Mobilekernelbench UNSUPPORTED set — the 38 ops ncnn does NOT natively support
(pnnx emits residual aten/prim OR a pnnx-only layer ncnn's runtime can't load).
These require the full Cand-kernel + GraphAgent pipeline. Frontier / stress set."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET          = ROOT / "dataset" / "Mobilekernelbench_unsupported"
MODEL            = "deepseek-v4-pro"
MAX_ROUNDS       = "15"
GRAPH_MAX_ROUNDS = "10"
PER_OP_TIMEOUT   = 1800            # 30 min hard cap per op
BACKENDS         = "base"
COMPILE_MODE     = "build_lib"
