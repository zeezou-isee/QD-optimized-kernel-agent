"""Export one op's OptimizeAgent run as a plot-friendly trace for paper figures.

Reads runs/<op>/backends/<backend>/optimize/summary.json (produced with
`--record-trace`) and emits a compact JSON with everything a figure needs:

  bd_axes        — how the BD grid is partitioned (axis names + value vocab)
  inner_config   — budgets (map/inner/coverage/patience, coarse grid settings)
  grid           — every covered niche -> best latency (the MAP-Elites heatmap),
                   with baseline/winner niches flagged
  rounds[]       — per QD round: cell, kept, param_space, the inner CLIMB
                   trajectory (grid then climb points, in eval order, each with
                   latency + stage), and the analytically PRUNED points + reasons

Without --record-trace the summary has no per-round trajectory/pruned/param_space;
this script says so and still emits the grid/axes it can.

Usage:
    python scripts/optimize_trace.py --op Dense_Convolution_2D
    python scripts/optimize_trace.py --op Conv --backend arm --out fig/conv_trace.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "opgen" / "runs"
RESULTS = REPO / "batch" / "results"


def _cell(c):
    return "/".join(map(str, c)) if isinstance(c, (list, tuple)) else c


def build(op: str, backend: str) -> dict:
    sp = RUNS / op / "backends" / backend / "optimize" / "summary.json"
    if not sp.exists():
        raise SystemExit(f"no optimize summary: {sp} (run run_optimize with --backend {backend})")
    s = json.loads(sp.read_text(encoding="utf-8"))
    ex = s.get("extra") or {}
    base_cell = _cell(ex.get("baseline_cell"))
    win_cell = _cell(ex.get("argmin_cell"))
    cells = (ex.get("archive") or {}).get("cells") or []
    grid = [{"cell": _cell(c.get("cell")), "latency_ms": c.get("latency_ms"),
             "is_baseline": _cell(c.get("cell")) == base_cell,
             "is_winner": _cell(c.get("cell")) == win_cell} for c in cells]

    rounds = []
    traced = 0
    for it in (ex.get("iterations") or []):
        r = {"round": it.get("round"), "cell": _cell(it.get("cell")),
             "directive": it.get("directive"), "kept": it.get("kept"),
             "cand_latency_ms": it.get("cand_latency"), "best_latency_ms": it.get("best_latency"),
             "evaluated": it.get("evaluated"), "pruned": it.get("pruned")}
        if "trajectory" in it:          # only present with --record-trace
            traced += 1
            r["param_space"] = it.get("param_space")
            r["best_params"] = it.get("best_params")
            r["trajectory"] = it.get("trajectory")        # [{point, latency_ms, stage, correct}]
            r["pruned_points"] = it.get("pruned_points")  # [{point, reason, stage}]
        rounds.append(r)

    return {
        "op": op, "backend": backend, "regime": ex.get("regime"),
        "has_trace": traced > 0,
        "bd_axes": ex.get("bd_axes"), "inner_config": ex.get("inner_config"),
        "baseline_cell": base_cell, "winner_cell": win_cell,
        "baseline_latency_ms": ex.get("baseline_latency_ms"),
        "best_latency_ms": (s.get("best_perf") or {}).get("min"),
        "coverage": ex.get("coverage"), "rounds_explored": ex.get("rounds"),
        "grid": grid, "rounds": rounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one op's optimize run as a plot-friendly trace.")
    ap.add_argument("--op", required=True)
    ap.add_argument("--backend", default="arm")
    ap.add_argument("--out", default=None, help="default batch/results/trace_<op>.json")
    args = ap.parse_args()

    t = build(args.op, args.backend)
    out = Path(args.out) if args.out else (RESULTS / f"trace_{args.op}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"op={t['op']} backend={t['backend']} regime={t['regime']}  has_trace={t['has_trace']}")
    if t["bd_axes"]:
        a1, a2 = t["bd_axes"]["axis1"], t["bd_axes"]["axis2"]
        print(f"  grid: {a1['name']}{a1['values']} × {a2['name']}{a2['values']}")
    print(f"  baseline_cell={t['baseline_cell']}  winner_cell={t['winner_cell']}  "
          f"coverage={t['coverage']}  rounds={t['rounds_explored']}")
    print(f"  covered bins: " + ", ".join(f"{g['cell']}={g['latency_ms']:.4g}"
          + ("⚑" if g["is_winner"] else "○" if g["is_baseline"] else "") for g in t["grid"]))
    if t["has_trace"]:
        npts = sum(len(r.get("trajectory") or []) for r in t["rounds"])
        npr = sum(len(r.get("pruned_points") or []) for r in t["rounds"])
        print(f"  TRACE: {sum(1 for r in t['rounds'] if 'trajectory' in r)} traced rounds, "
              f"{npts} measured points, {npr} pruned points")
    else:
        print("  (no per-round trajectory — re-run the op with --record-trace to capture climb/pruning)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
