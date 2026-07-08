"""Roll up OptimizeAgent results across a dataset's ops into one table.

Reads each op's optimize summary (runs/<op>/backends/<backend>/optimize/summary.json)
and records the QD search outcome — ROUNDS, coverage, regime, baseline vs best
real-phone latency, self-speedup, stop reason — plus a reliability flag:

  real     — a trustworthy ms-scale win/tie (baseline above the noise floor)
  suspect  — μs-scale (baseline < noise floor) → any "speedup" is timer noise
  tainted  — implausible speedup (> cap) → degenerate winner / device path-mixing
  crash    — optimize did not produce a summary (e.g. arm kernel returned -100)

Usage:
    python scripts/optimize_rollup.py                       # v2 dataset, arm
    python scripts/optimize_rollup.py --backend base
    python scripts/optimize_rollup.py --ops Conv,LayerNorm
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "opgen" / "runs"
RESULTS = REPO / "batch" / "results"
DATASET = REPO / "dataset" / "Mobilekernelbench_optimized"

NOISE_FLOOR_MS = 0.02   # below this the device timer is unreliable
SPEEDUP_CAP = 8.0       # above this a "win" is almost surely degenerate / path-mix


def _ops(explicit: str | None) -> list[tuple[str, str]]:
    """(op, category) list. From selection.json if present, else *.py scan."""
    sel = DATASET / "selection.json"
    if explicit:
        return [(o.strip(), "") for o in explicit.split(",") if o.strip()]
    if sel.exists():
        return [(r["op"], r.get("category", "")) for r in json.loads(sel.read_text())]
    return [(p.stem, p.parent.name) for p in sorted(DATASET.rglob("*.py"))]


def row_for(op: str, category: str, backend: str) -> dict:
    sp = RUNS / op / "backends" / backend / "optimize" / "summary.json"
    row = {"op": op, "category": category, "backend": backend, "status": "crash",
           "regime": None, "rounds": None, "coverage": None,
           "kept_rounds": None, "improved": None,
           "baseline_ms": None, "best_ms": None, "self_speedup": None,
           "stopped_reason": None, "flag": "crash"}
    if not sp.exists():
        return row
    try:
        s = json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return row
    ex = s.get("extra") or {}
    best = s.get("best_perf") or {}
    base = ex.get("baseline_latency_ms")
    bestv = best.get("min") if best.get("min") is not None else best.get("avg")
    spd = (base / bestv) if (isinstance(base, (int, float)) and isinstance(bestv, (int, float)) and bestv) else None
    # best_round is a binary flag in the map_elites path (0=improved, -1=baseline
    # kept), NOT the winning round index. The real signal is how many QD rounds
    # produced a NEW best (iterations with kept=True).
    its = ex.get("iterations") or []
    kept = sum(1 for i in its if i.get("kept"))
    improved = s.get("best_round") not in (None, -1)
    row.update(status="success", regime=ex.get("regime"), rounds=ex.get("rounds"),
               coverage=ex.get("coverage"), kept_rounds=kept, improved=improved,
               baseline_ms=base, best_ms=bestv,
               self_speedup=round(spd, 4) if spd else None,
               stopped_reason=s.get("stopped_reason"))
    # reliability flag
    if isinstance(spd, (int, float)) and spd > SPEEDUP_CAP:
        row["flag"] = "tainted"
    elif isinstance(base, (int, float)) and base < NOISE_FLOOR_MS:
        row["flag"] = "suspect"
    else:
        row["flag"] = "real"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll up OptimizeAgent results (rounds + speedup).")
    ap.add_argument("--backend", default="arm")
    ap.add_argument("--ops", default=None, help="comma list (else the v2 dataset)")
    ap.add_argument("--out-csv", default=str(RESULTS / "optimize_rollup.csv"))
    ap.add_argument("--out-md", default=str(RESULTS / "optimize_rollup.md"))
    args = ap.parse_args()

    rows = [row_for(op, cat, args.backend) for op, cat in _ops(args.ops)]

    # write CSV
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    real = [r for r in rows if r["flag"] == "real" and isinstance(r["self_speedup"], (int, float))]
    xs = [r["self_speedup"] for r in real]
    buckets = {b: [r for r in rows if r["flag"] == b] for b in ("real", "suspect", "tainted", "crash")}

    md = [f"# OptimizeAgent rollup — {DATASET.name} ({args.backend})\n"]
    md.append(f"{len(rows)} ops · real={len(buckets['real'])} suspect={len(buckets['suspect'])} "
              f"tainted={len(buckets['tainted'])} crash={len(buckets['crash'])}\n")
    if xs:
        md.append(f"real-win self-speedup: median **{statistics.median(xs):.3f}×** "
                  f"mean {statistics.mean(xs):.3f}× max {max(xs):.3f}× · "
                  f"improved(>1.02×) {sum(1 for x in xs if x > 1.02)}/{len(xs)}\n")
    md.append("\nflags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · "
              "tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)\n")
    md.append("\n`rounds` = QD candidates explored · `kept` = rounds that set a new best "
              "(the actual optimization steps). A win means best_kernel is a *different* "
              "LLM-varied + param-tuned kernel that measured faster on the phone.\n")
    md.append("\n| op | cat | regime | rounds | kept | cov | baseline_ms | best_ms | self_speedup | flag | stopped |")
    md.append("|----|-----|--------|-------:|-----:|----:|------------:|--------:|-------------:|------|---------|")
    order = {"real": 0, "tainted": 1, "suspect": 2, "crash": 3}
    for r in sorted(rows, key=lambda r: (order[r["flag"]],
                    -(r["self_speedup"] if isinstance(r["self_speedup"], (int, float)) else 0))):
        bm = f"{r['baseline_ms']:.3f}" if isinstance(r["baseline_ms"], (int, float)) else "—"
        be = f"{r['best_ms']:.3f}" if isinstance(r["best_ms"], (int, float)) else "—"
        sp = f"{r['self_speedup']:.3f}" if isinstance(r["self_speedup"], (int, float)) else "—"
        md.append(f"| `{r['op']}` | {r['category']} | {r['regime'] or '—'} | "
                  f"{r['rounds'] if r['rounds'] is not None else '—'} | "
                  f"{r['kept_rounds'] if r['kept_rounds'] is not None else '—'} | "
                  f"{r['coverage'] if r['coverage'] is not None else '—'} | {bm} | {be} | {sp} | "
                  f"{r['flag']} | {r['stopped_reason'] or '—'} |")
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\ncsv -> {out_csv}\nmd  -> {args.out_md}")


if __name__ == "__main__":
    main()
