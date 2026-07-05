"""Behavior descriptors — two bottleneck-conditional coordinate systems (Workflow §4.2).

BD answers "哪一类", NOT "多好" (fitness). 拿延迟当 BD 就废了。 The two main axes of
each coordinate system are **structural labels known at generation time** (§4.3),
so a candidate can be located into its niche BEFORE the inner search runs — which
is what lets the outer loop decide whether a niche is worth spending budget on.

We compute the cell from the template's structural tags (`techniques`), not by
running the kernel — 不让 LLM/实测 来猜 BD (§4.3 档案污染防护).

Coordinate A (memory_bound):  axis1 = 数据布局/访存模式族, axis2 = 分块策略族
Coordinate B (compute_bound): axis1 = 算法族,             axis2 = 计算映射/指令类
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .roofline import COMPUTE_BOUND, MEMORY_BOUND

# axis vocabularies (the niche grid is the cartesian product of axis1 × axis2).
# These are the HARDCODED FALLBACK — the source of truth is now the externalized
# Σ registry (policy/sigma.py + experience_pool/wiki/sigma/<backend>.json), which
# `classify_with_novelty` loads. `axes()`/`classify()` keep using these tuples so
# the legacy API + tests stay green with zero config; the Σ-aware path
# (classify_with_novelty) supersedes them when a wiki_root is available.
_A_LAYOUT = ("nchw", "nhwc", "packed")
_A_TILING = ("none", "single", "double")
_B_ALGO = ("direct", "gemm", "winograd", "fft", "dw")
_B_MAPPING = ("scalar", "vec", "dotprod")


def axes(regime: str) -> tuple[tuple[str, tuple], tuple[str, tuple]]:
    """Return ((axis1_name, axis1_values), (axis2_name, axis2_values)) for a regime."""
    if regime == COMPUTE_BOUND:
        return ("algo_family", _B_ALGO), ("compute_mapping", _B_MAPPING)
    return ("layout_family", _A_LAYOUT), ("tiling_strategy", _A_TILING)


def _has(tags: list[str], *needles: str) -> bool:
    blob = " ".join(t.lower() for t in tags)
    return any(n in blob for n in needles)


def _classify_memory(tags: list[str]) -> tuple[str, str]:
    if _has(tags, "nc4hw4", "pack", "packed"):
        layout = "packed"
    elif _has(tags, "nhwc"):
        layout = "nhwc"
    else:
        layout = "nchw"
    if _has(tags, "double", "register", "two-level", "2-level"):
        tiling = "double"
    elif _has(tags, "tile", "tiling", "block", "blocking"):
        tiling = "single"
    else:
        tiling = "none"
    return layout, tiling


def _classify_compute(tags: list[str]) -> tuple[str, str]:
    if _has(tags, "winograd"):
        algo = "winograd"
    elif _has(tags, "im2col", "gemm", "sgemm"):
        algo = "gemm"
    elif _has(tags, "fft"):
        algo = "fft"
    elif _has(tags, "depthwise", "dw"):
        algo = "dw"
    else:
        algo = "direct"
    if _has(tags, "dotprod", "sdot", "udot"):
        mapping = "dotprod"
    elif _has(tags, "neon", "vec", "simd", "vectoriz", "cooperative", "vec4"):
        mapping = "vec"
    else:
        mapping = "scalar"
    return algo, mapping


def classify(techniques: list[str], regime: str) -> tuple[str, str]:
    """Map a template's structural tags onto its (axis1, axis2) niche cell.

    Legacy keyword-only path (no Σ, no novelty). Kept for the existing callers
    (baseline elite, tests). New code should prefer `classify_with_novelty`,
    which is Σ-driven and reports out-of-vocabulary structural labels.
    """
    tags = list(techniques or [])
    if regime == COMPUTE_BOUND:
        return _classify_compute(tags)
    return _classify_memory(tags)


def grid_size(regime: str) -> int:
    (_, a1), (_, a2) = axes(regime)
    return len(a1) * len(a2)


def classify_with_novelty(
    techniques: list[str],
    regime: str,
    *,
    backend: str = "base",
    bd_labels: dict[str, str] | None = None,
    wiki_root: Path | str | None = None,
) -> tuple[tuple[str, str], dict[str, str]]:
    """Σ-aware classification with axis-extension detection (Method M2.4/M2.5.2).

    Returns ((axis1_value, axis2_value), novel) where `novel` maps
    axis_name -> proposed_value for any axis whose value is OUTSIDE the current
    Σ(backend, regime) vocabulary — i.e. the LLM proposed a genuinely new
    structural label. Such a candidate can still be located (it OPENS a new
    niche at its declared coordinate); if it later wins that niche it feeds the
    axis-extension write-back (map_elites → sigma.record_win).

    Coordinate resolution per axis:
      1. explicit `bd_labels[axis_name]` if the LLM declared it —
         in-Σ value → use as-is; out-of-Σ value → use as-is + flag novel.
      2. else keyword-classify from `techniques` against Σ's keyword map
         (always yields an in-Σ value; never novel).

    Falls back to the hardcoded vocab (via sigma.load's synthesized Σ) when no
    wiki_root / no JSON — so this never crashes and matches `classify()` when the
    LLM gave no explicit labels.
    """
    from . import sigma as _sigma  # local import avoids import cycle at module load

    tags = list(techniques or [])
    labels = dict(bd_labels or {})

    if wiki_root is not None:
        sg = _sigma.load(wiki_root, backend)
    else:
        # No wiki root: synthesize Σ from the hardcoded fallback so the keyword
        # map still exists. (sigma.load with a nonexistent path does exactly this.)
        sg = _sigma.load(Path("/nonexistent-wiki-root"), backend)

    cell: list[str] = []
    novel: dict[str, str] = {}
    for which in ("axis1", "axis2"):
        axis_name = sg.axis_name(regime, which)
        declared = labels.get(axis_name) or labels.get(which)
        if declared:
            declared = str(declared).strip().lower()
            cell.append(declared)
            if not sg.is_known(regime, which, declared):
                novel[axis_name] = declared
        else:
            cell.append(sg.classify_axis(regime, which, tags))
    return (cell[0], cell[1]), novel
