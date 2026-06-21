"""Local hill climb / coordinate descent (Workflow §6.1 step ③).

Exploits parameter-layer local smoothness (主文档 §4.1): from the best coarse
point, step ±1 index along each axis and move whenever a neighbor is faster.
Cliffs (illegal / out-of-cache points) are kept out by the constraint engine
upstream, so the climb never falls into an illegal region.

The climb is generic over "how to measure a point": the caller passes a
`measure(point) -> (latency_or_None)` callback that already folds in
correctness + feasibility (None = unusable point, skip).
"""

from __future__ import annotations

from typing import Any, Callable

from schemas import ParamSpec


def _neighbors(point: dict[str, Any], params: dict[str, ParamSpec]) -> list[dict[str, Any]]:
    """All points one index step away along a single axis."""
    out: list[dict[str, Any]] = []
    for name, spec in params.items():
        vals = spec.values
        try:
            i = vals.index(point[name])
        except (ValueError, KeyError):
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                nb = dict(point)
                nb[name] = vals[j]
                out.append(nb)
    return out


def hill_climb(
    start: dict[str, Any],
    start_latency: float,
    params: dict[str, ParamSpec],
    measure: Callable[[dict[str, Any]], float | None],
    *,
    budget: int = 6,
) -> tuple[dict[str, Any], float, int]:
    """Coordinate-descent from `start`. Returns (best_point, best_latency, used).

    `used` counts how many real measurements were consumed (caller budget bookkeeping).
    Stops on: no improving neighbor, or budget exhausted.
    """
    best_pt, best_lat = dict(start), start_latency
    used = 0
    seen = {tuple(sorted(start.items()))}

    while used < budget:
        improved = False
        for nb in _neighbors(best_pt, params):
            key = tuple(sorted(nb.items()))
            if key in seen:
                continue
            if used >= budget:
                break
            seen.add(key)
            lat = measure(nb)
            used += 1
            if lat is not None and lat < best_lat:
                best_pt, best_lat = nb, lat
                improved = True
                break          # greedy: restart neighborhood from the new best
        if not improved:
            break
    return best_pt, best_lat, used
