"""Coarse grid enumeration (Workflow §6.1 step ②).

Pick a few representative values per axis (low / mid / high) and take their
cartesian product, so the inner search can locate the basin with a handful of
measurements before hill-climbing. We do NOT exhaustively enumerate the full
space — that's the whole point of "coarse".
"""

from __future__ import annotations

import itertools
from typing import Any

from schemas import ParamSpec


def _representatives(values: list[Any], k: int = 3) -> list[Any]:
    """Up to k spread-out values: first, middle, last (dedup, order-preserving)."""
    n = len(values)
    if n <= k:
        return list(values)
    if k <= 1:
        return [values[0]]
    idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [values[i] for i in idxs]


def coarse_points(params: dict[str, ParamSpec], per_axis: int = 3,
                  max_points: int = 12) -> list[dict[str, Any]]:
    """Cartesian product of per-axis representatives, capped at max_points.

    Empty params → a single empty point (the kernel has no knobs to tune; the
    inner search still measures it once for a baseline-equivalent latency).
    """
    if not params:
        return [{}]
    axes = {name: _representatives(spec.values, per_axis) for name, spec in params.items()}
    names = list(axes)
    points: list[dict[str, Any]] = []
    for combo in itertools.product(*(axes[n] for n in names)):
        points.append(dict(zip(names, combo)))
        if len(points) >= max_points:
            break
    return points
