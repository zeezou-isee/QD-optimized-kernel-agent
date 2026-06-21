"""Experience pool — the 兵器谱 (Workflow §8.4 / 主文档 §8).

A persistent, cross-task / cross-device store of winning kernels, keyed by
**regime** (memory_bound / compute_bound). Two jobs:

  1. **Seed (地板, 不过滤)**: when starting a new operator, drop same-regime known
     kernels into the archive as initial elites — even a slow seed survives if it
     occupies an empty niche (§7.1 情形A). Cross-operator reuse: the pool is
     same-regime, NOT same-operator (§7.1: "compute-bound 该有哪些算法族" 照样可用).

  2. **Persist**: after a run, fold the whole archive back in, so the pool gets
     richer the more you use it (越用越富).

Storage = a single JSON file (list of records). Each record carries the regime,
the niche cell, the kernel source, params, techniques, and the measured latency.
Hardware id is recorded for provenance / weak cross-device priors (M3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .archive import Archive, Elite


@dataclass
class PoolRecord:
    regime: str
    cell: list
    latency_ms: float
    kernel_code: dict[str, str]
    params: dict[str, Any] = field(default_factory=dict)
    techniques: list[str] = field(default_factory=list)
    op_class: str = ""
    hardware: str = ""

    def to_elite(self) -> Elite:
        return Elite(cell=tuple(self.cell), latency_ms=self.latency_ms,
                     kernel_code=self.kernel_code, params=self.params,
                     techniques=self.techniques, source="seed")


class ExperiencePool:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[PoolRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.records = [PoolRecord(**r) for r in raw.get("records", [])]
            except Exception:  # noqa: BLE001 — corrupt pool shouldn't crash a run
                self.records = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"records": [asdict(r) for r in self.records]},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    # --- seed (地板, 不过滤) -------------------------------------------------
    def seeds_for(self, regime: str, *, hardware: str | None = None) -> list[Elite]:
        """Same-regime seed elites (floor). Same-hardware first, then others as
        weak cross-device priors. NO filtering — slow seeds still占空 niche."""
        same = [r for r in self.records if r.regime == regime]
        if hardware is not None:
            same.sort(key=lambda r: 0 if r.hardware == hardware else 1)
        return [r.to_elite() for r in same]

    # --- persist ------------------------------------------------------------
    def add_archive(self, archive: Archive, *, regime: str, op_class: str = "",
                    hardware: str = "") -> int:
        """Fold every elite of an archive into the pool. Returns #records added."""
        added = 0
        for e in archive.elites():
            self.records.append(PoolRecord(
                regime=regime, cell=list(e.cell), latency_ms=e.latency_ms,
                kernel_code=e.kernel_code, params=e.params,
                techniques=e.techniques, op_class=op_class, hardware=hardware))
            added += 1
        return added
