"""Sigma (Σ) — the machine-readable optimization-space axis registry.

This is the reification of the paper's central object `Σ(h,L)` (Method M0.1/M2.5):
the per-backend, regime-scoped set of structural AXES with their enumerated
value domains. Historically these lived as hardcoded tuples in `bd.py`
(`_A_LAYOUT`, `_B_ALGO`, …) — knowledge, but neither externally readable nor
growable. This module makes Σ:

  1. **machine-readable**  — a JSON file per backend under
     `experience_pool/wiki/sigma/<backend>.json`;
  2. **growable (axis-extension, M2.5.2)** — `record_win()` accumulates a
     cross-task counter for structural labels the LLM proposed that were NOT in
     Σ; once a novel label wins/opens a niche `N_promote` times across tasks, it
     is promoted into the axis's `values` and written back — `Σ` literally grows.

Graceful degradation: if the JSON is missing/corrupt, `load()` synthesizes Σ
from the `bd.py` hardcoded defaults, so the QD loop and existing tests keep
working with zero config. Writes are atomic (tmp + os.replace).

JSON schema (version 1):
{
  "backend": "arm",
  "version": 1,
  "regimes": {
    "memory_bound": {
      "axis1": {"name": "layout_family",
                "values": ["nchw","nhwc","packed"],
                "keywords": {"packed": ["nc4hw4","pack","packed"], "nhwc": ["nhwc"]}},
      "axis2": {"name": "tiling_strategy", "values": [...], "keywords": {...}}
    },
    "compute_bound": { "axis1": {...}, "axis2": {...} }
  },
  "pending": {                              # axis-extension counters (cross-task)
    "compute_bound|algo_family|strassen": {"wins": 2, "first_seen_task": "MatMul", ...}
  },
  "promoted": [                             # audit log of what grew Σ + when
    {"regime": "compute_bound", "axis": "algo_family", "value": "strassen",
     "wins": 3, "tasks": ["MatMul","MatMul_3d","Gemm"]}
  ]
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# regime keys — imported lazily to avoid a bd<->sigma import cycle
_MEMORY_BOUND = "memory_bound"
_COMPUTE_BOUND = "compute_bound"

DEFAULT_N_PROMOTE = 3

# Hardcoded fallback Σ — mirrors bd.py's historical tuples. Used to synthesize a
# JSON when none exists on disk. `keywords` reproduce bd.py::_classify_* mapping
# so the externalized Σ classifies identically to the old hardcoded path.
_FALLBACK_SIGMA: dict[str, Any] = {
    "regimes": {
        _MEMORY_BOUND: {
            "axis1": {
                "name": "layout_family",
                "values": ["nchw", "nhwc", "packed"],
                "keywords": {
                    "packed": ["nc4hw4", "pack", "packed"],
                    "nhwc": ["nhwc"],
                    "nchw": [],          # default bucket
                },
            },
            "axis2": {
                "name": "tiling_strategy",
                "values": ["none", "single", "double"],
                "keywords": {
                    "double": ["double", "register", "two-level", "2-level"],
                    "single": ["tile", "tiling", "block", "blocking"],
                    "none": [],
                },
            },
        },
        _COMPUTE_BOUND: {
            "axis1": {
                "name": "algo_family",
                "values": ["direct", "gemm", "winograd", "fft", "dw"],
                "keywords": {
                    "winograd": ["winograd"],
                    "gemm": ["im2col", "gemm", "sgemm"],
                    "fft": ["fft"],
                    "dw": ["depthwise", "dw"],
                    "direct": [],
                },
            },
            "axis2": {
                "name": "compute_mapping",
                "values": ["scalar", "vec", "dotprod"],
                "keywords": {
                    "dotprod": ["dotprod", "sdot", "udot"],
                    "vec": ["neon", "vec", "simd", "vectoriz", "cooperative", "vec4"],
                    "scalar": [],
                },
            },
        },
    },
    "pending": {},
    "promoted": [],
}


def sigma_path(wiki_root: Path | str, backend: str) -> Path:
    return Path(wiki_root) / "sigma" / f"{backend}.json"


@dataclass
class Sigma:
    """In-memory view of one backend's Σ, with load/mutate/save."""
    backend: str
    path: Path
    data: dict[str, Any]

    # -------------------------------------------------------------- queries
    def _axis(self, regime: str, which: str) -> dict[str, Any] | None:
        reg = self.data.get("regimes", {}).get(regime)
        if not reg:
            return None
        return reg.get(which)

    def axis_name(self, regime: str, which: str) -> str:
        a = self._axis(regime, which)
        return a.get("name", which) if a else which

    def values(self, regime: str, which: str) -> list[str]:
        a = self._axis(regime, which)
        return list(a.get("values", [])) if a else []

    def keywords(self, regime: str, which: str) -> dict[str, list[str]]:
        a = self._axis(regime, which)
        return dict(a.get("keywords", {})) if a else {}

    def is_known(self, regime: str, which: str, value: str) -> bool:
        return value in self.values(regime, which)

    def classify_axis(self, regime: str, which: str, tags: list[str]) -> str:
        """Keyword-match an axis value from free technique tags (mirrors the old
        bd.py behavior, but driven by Σ's keyword map). Returns the last value
        (the default bucket, whose keyword list is empty) when nothing matches."""
        blob = " ".join(t.lower() for t in (tags or []))
        vals = self.values(regime, which)
        kw = self.keywords(regime, which)
        for v in vals:
            needles = kw.get(v, [])
            if needles and any(n in blob for n in needles):
                return v
        # default bucket = the value whose keyword list is empty, else last value
        for v in vals:
            if not kw.get(v):
                return v
        return vals[-1] if vals else "unknown"

    # -------------------------------------------------- axis-extension (grow)
    def record_win(self, regime: str, axis_name: str, value: str, *,
                   task: str = "", n_promote: int = DEFAULT_N_PROMOTE) -> dict | None:
        """Register that a NOVEL structural label `value` (not in Σ[regime][axis])
        won/opened a niche on task `task`. Accumulates a cross-task counter; when
        it reaches `n_promote` DISTINCT tasks, promote `value` into the axis's
        `values` and append a `promoted` audit record.

        Returns a promotion-event dict when a promotion happens this call, else
        None. Idempotent per (regime,axis,value,task): the same task only counts
        once toward promotion.
        """
        # resolve which axis slot (axis1/axis2) this axis_name refers to
        which = None
        for w in ("axis1", "axis2"):
            if self.axis_name(regime, w) == axis_name:
                which = w
                break
        if which is None:
            return None
        if self.is_known(regime, which, value):
            return None  # already stable — nothing to grow

        key = f"{regime}|{axis_name}|{value}"
        pending = self.data.setdefault("pending", {})
        rec = pending.setdefault(key, {"wins": 0, "tasks": [], "which": which})
        t = task or "?"
        if t not in rec["tasks"]:
            rec["tasks"].append(t)
        rec["wins"] = len(rec["tasks"])

        if rec["wins"] >= n_promote:
            # PROMOTE: Σ grows.
            axis = self._axis(regime, which)
            if axis is not None and value not in axis["values"]:
                axis["values"].append(value)
            event = {"regime": regime, "axis": axis_name, "value": value,
                     "wins": rec["wins"], "tasks": list(rec["tasks"])}
            self.data.setdefault("promoted", []).append(event)
            pending.pop(key, None)
            return event
        return None

    def render_for_prompt(self, regime: str) -> str:
        """Human-readable axis menu for the given regime, to inject into the
        proposer prompt (Method M2.4: the LLM projects onto these axes). Lists
        each axis name + its current enumerated value domain. Returns "" if the
        regime is unknown to this Σ."""
        reg = self.data.get("regimes", {}).get(regime)
        if not reg:
            return ""
        lines = [f"Backend `{self.backend}`, regime `{regime}`. "
                 f"Pick bd_labels values from these menus (or declare a new one to extend):"]
        for which in ("axis1", "axis2"):
            a = reg.get(which)
            if not a:
                continue
            vals = ", ".join(f"`{v}`" for v in a.get("values", []))
            lines.append(f"- **{a.get('name', which)}**: {{{vals}}}")
        return "\n".join(lines)

    def size(self) -> int:
        """Total number of enumerated structural axis values across all regimes —
        the |Σ| metric for the paper's Σ-growth curve (Figure E)."""
        n = 0
        for reg in self.data.get("regimes", {}).values():
            for which in ("axis1", "axis2"):
                n += len(reg.get(which, {}).get("values", []))
        return n

    # ---------------------------------------------------------------- io
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)


def load(wiki_root: Path | str, backend: str) -> Sigma:
    """Load Σ for a backend. Synthesizes from the hardcoded fallback when the
    JSON is missing or corrupt (does NOT write it — caller decides when to
    persist; a read-only ablation should not create files)."""
    p = sigma_path(wiki_root, backend)
    data: dict[str, Any]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "regimes" not in data:
            raise ValueError("missing 'regimes'")
    except (OSError, ValueError, json.JSONDecodeError):
        data = {"backend": backend, "version": 1, **json.loads(json.dumps(_FALLBACK_SIGMA))}
    data.setdefault("backend", backend)
    data.setdefault("version", 1)
    data.setdefault("pending", {})
    data.setdefault("promoted", [])
    return Sigma(backend=backend, path=p, data=data)


def fallback_data(backend: str) -> dict[str, Any]:
    """The hardcoded Σ as a fresh dict (used to seed a backend's JSON on disk)."""
    return {"backend": backend, "version": 1,
            **json.loads(json.dumps(_FALLBACK_SIGMA))}
