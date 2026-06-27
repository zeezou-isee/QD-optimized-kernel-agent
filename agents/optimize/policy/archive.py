"""MAP-Elites archive (Workflow §7 / 主文档 附录A).

每个 cell (行为 niche) 只存一个精英 = 该行为下找到的最快 kernel。竞争**只在格子
内部** (★局部竞争★) —— 这是"构造性保多样"的要害: 一个新颖但稍慢的 kernel 只要落在
空格子就存活, 不会被全局最优挤掉, 从机制上根除"过拟合到已知配方"。

Cell competition uses a noise floor σ (§5.3): a candidate replaces the incumbent
only if it is faster by more than σ, so measurement noise can't drive a false swap.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Elite:
    cell: tuple                       # niche coordinates (axis1, axis2)
    latency_ms: float
    kernel_code: dict[str, str]       # materialized (compilable) source
    params: dict[str, Any] = field(default_factory=dict)
    techniques: list[str] = field(default_factory=list)
    source: str = "search"            # "seed" | "search"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell"] = list(self.cell)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Elite":
        return cls(cell=tuple(d["cell"]), latency_ms=d["latency_ms"],
                   kernel_code=d.get("kernel_code", {}), params=d.get("params", {}),
                   techniques=d.get("techniques", []), source=d.get("source", "search"))


class Archive:
    """cell -> Elite, with per-cell visit counts for novelty pressure."""

    def __init__(self) -> None:
        self.cells: dict[tuple, Elite] = {}
        self.visits: dict[tuple, int] = {}

    # --- mutation ----------------------------------------------------------
    def place(self, elite: Elite, sigma: float = 0.0) -> bool:
        """Cell competition. Returns True if `elite` became (or stayed) the cell's
        elite, i.e. the cell was empty or `elite` is faster by more than σ."""
        self.visits[elite.cell] = self.visits.get(elite.cell, 0) + 1
        cur = self.cells.get(elite.cell)
        if cur is None or elite.latency_ms < cur.latency_ms - sigma:
            self.cells[elite.cell] = elite
            return True
        return False

    # --- queries -----------------------------------------------------------
    def coverage(self) -> int:
        return len(self.cells)

    def argmin(self) -> Elite | None:
        if not self.cells:
            return None
        return min(self.cells.values(), key=lambda e: e.latency_ms)

    def elites(self) -> list[Elite]:
        return list(self.cells.values())

    def select_parents(self, k: int = 1, *, w_novelty: float = 0.4,
                       rng: random.Random | None = None) -> list[Elite]:
        """Quality-biased + novelty sampling (§7.2 step①).

        score = quality (best_latency / elite_latency ∈ (0,1]) + w_novelty *
        novelty (1/(1+visits[cell]) ∈ (0,1]). Sampled WITHOUT replacement,
        weighted by score, so good *and* under-explored niches both get picked
        (avoids collapsing onto the single fastest elite). Deterministic given rng.
        """
        pool = self.elites()
        if not pool:
            return []
        rng = rng or random.Random(0)
        best = min(e.latency_ms for e in pool)
        chosen: list[Elite] = []
        remaining = list(pool)
        for _ in range(min(k, len(remaining))):
            weights = []
            for e in remaining:
                quality = best / e.latency_ms if e.latency_ms > 0 else 1.0
                novelty = 1.0 / (1.0 + self.visits.get(e.cell, 0))
                weights.append(max(1e-9, quality + w_novelty * novelty))
            pick = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
            chosen.append(remaining.pop(pick))
        return chosen

    def empty_cell_count(self, regime: str) -> int:
        from .bd import grid_size
        return max(0, grid_size(regime) - self.coverage())

    # --- persistence (兵器谱) ----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"cells": [e.to_dict() for e in self.cells.values()],
                "visits": [[list(c), n] for c, n in self.visits.items()]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Archive":
        a = cls()
        for ed in d.get("cells", []):
            e = Elite.from_dict(ed)
            a.cells[e.cell] = e
        for c, n in d.get("visits", []):
            a.visits[tuple(c)] = n
        return a
