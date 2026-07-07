"""Roll-up statistics across operators — compile / functional / speedup.

Aggregates the THREE metrics that live in separate places today into one per-op
table + summary rates:

  1. compile correctness   — runs/<op>/operator/summary.json phases.production.compile.ok
                             (SEPARATED from correctness — batch all.json fuses both
                             into a single `production` bool; this un-fuses them)
  2. functional correctness — phases.kernel.status (numeric vs PyTorch),
                             phases.end_to_end_numeric.passed (whole ncnn::Net vs
                             PyTorch), phases.production.correctness.passed
  3. speedup vs ncnn native — batch/results/perf_compare.json (speedup_shipped /
                             speedup_fair), joined by op:backend. Produced by
                             opgen/cli/run_perf_compare.py.

Ground truth is each op's own summary.json under runs/ (the batch result JSONs
conflate compile+correctness). Speedup is merged in from perf_compare.json.

Usage:
    # roll up the 190-op operator run, join arm speedups, print + write CSV/JSON
    python scripts/rollup_stats.py --source batch/results/all.json --backend arm

    # roll up an explicit op list
    python scripts/rollup_stats.py --ops Abs,Conv,Gemm --backend arm

    # scan every op that has an operator/summary.json
    python scripts/rollup_stats.py

Outputs (default under batch/results/):
    rollup.csv           — one row per op (the three metrics, un-fused)
    rollup_summary.json  — aggregate rates + speedup stats
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


def _dig(d: dict, *path, default=None):
    """Safe nested get: _dig(s, 'phases', 'production', 'compile', 'ok')."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def op_row(op: str, runs_root: Path, perf: dict, backend: str) -> dict:
    """Build one op's un-fused metric row from its operator summary + perf join."""
    row = {"op": op, "operator_run": False, "status": None,
           "already_in_ncnn": None,
           "kernel_numeric": None, "kernel_max_diff": None,
           "e2e": None, "e2e_max_diff": None,
           "compile": None, "correctness": None, "correctness_max_diff": None,
           "device_status": None, "device_latency": None, "device_speedup": None,
           "native_supported": None,
           "speedup_shipped": None, "speedup_fair": None,
           "ours_ms": None, "native_shipped_ms": None, "native_fair_ms": None}

    sj = runs_root / op / "operator" / "summary.json"
    if sj.exists():
        try:
            s = json.loads(sj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            s = {}
        if s:
            row["operator_run"] = True
            row["status"] = s.get("status")
            row["already_in_ncnn"] = _dig(s, "phases", "existence_check", "already_in_ncnn")
            row["kernel_numeric"] = _dig(s, "phases", "kernel", "status")
            row["kernel_max_diff"] = _dig(s, "phases", "kernel", "max_diff")
            row["e2e"] = _dig(s, "phases", "end_to_end_numeric", "passed")
            row["e2e_max_diff"] = _dig(s, "phases", "end_to_end_numeric", "max_diff")
            # THE un-fusing: compile and correctness are two separate booleans.
            row["compile"] = _dig(s, "phases", "production", "compile", "ok")
            row["correctness"] = _dig(s, "phases", "production", "correctness", "passed")
            row["correctness_max_diff"] = _dig(s, "phases", "production", "correctness", "max_diff")
            # device-in-the-loop status from the backend-appropriate kernel phase
            _kphase = "kernel_arm" if backend == "arm" else "kernel"
            row["device_status"] = _dig(s, "phases", _kphase, "device_status")
            row["device_latency"] = _dig(s, "phases", _kphase, "device_latency")
            row["device_speedup"] = _dig(s, "phases", _kphase, "device_speedup")

    p = perf.get(f"{op}:{backend}") or {}
    if p:
        row["native_supported"] = p.get("native_supported")
        row["speedup_shipped"] = p.get("speedup_shipped")
        row["speedup_fair"] = p.get("speedup_fair")
        row["ours_ms"] = _dig(p, "ours_shipped", "latency_min") or _dig(p, "ours", "gpu_latency_min_ms")
        row["native_shipped_ms"] = _dig(p, "native_shipped", "latency_min")
        row["native_fair_ms"] = _dig(p, "native_fair", "latency_min")
    return row


def _rate(vals: list, pred) -> tuple[int, int]:
    """(#matching, #considered-non-None) for a predicate over a metric column."""
    considered = [v for v in vals if v is not None]
    return sum(1 for v in considered if pred(v)), len(considered)


def summarize(rows: list[dict]) -> dict:
    def col(k):
        return [r[k] for r in rows]

    def _pct(k, pred):
        n, d = _rate(col(k), pred)
        return {"pass": n, "of": d, "rate": round(n / d, 4) if d else None}

    truthy = lambda v: v is True  # noqa: E731
    ok_status = lambda v: str(v).lower() in ("success", "passed", "ok")  # noqa: E731

    def _speedup_stats(k):
        xs = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
                "median": round(statistics.median(xs), 3),
                "min": round(min(xs), 3), "max": round(max(xs), 3),
                "n_faster_than_native": sum(1 for x in xs if x > 1.0)}

    # device-in-the-loop: passed vs (passed+failed) among ops the gate actually ran
    dev = [r["device_status"] for r in rows if r["device_status"] in ("passed", "failed")]
    device_gate = {"passed": sum(1 for v in dev if v == "passed"), "ran": len(dev),
                   "rate": round(sum(1 for v in dev if v == "passed") / len(dev), 4) if dev else None,
                   "skipped": sum(1 for r in rows if r["device_status"] == "skipped")}
    return {
        "n_ops": len(rows),
        "n_operator_runs": sum(1 for r in rows if r["operator_run"]),
        "compile_correctness": _pct("compile", truthy),
        "kernel_numeric": _pct("kernel_numeric", ok_status),
        "e2e_numeric": _pct("e2e", truthy),
        "production_correctness": _pct("correctness", truthy),
        "device_gate": device_gate,
        "device_speedup_inline": _speedup_stats("device_speedup"),
        "already_in_ncnn": _pct("already_in_ncnn", truthy),
        "speedup_shipped": _speedup_stats("speedup_shipped"),
        "speedup_fair": _speedup_stats("speedup_fair"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll up compile/functional/speedup stats across ops.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--source", default=None,
                     help="a batch result JSON (e.g. batch/results/all.json) whose keys "
                          "define the op set")
    src.add_argument("--ops", default=None, help="explicit comma list of ops")
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--backend", default="arm", help="backend to join perf_compare on")
    ap.add_argument("--perf", default=str(RESULTS / "perf_compare.json"),
                    help="perf_compare.json to merge speedups from")
    ap.add_argument("--out-csv", default=str(RESULTS / "rollup.csv"))
    ap.add_argument("--out-json", default=str(RESULTS / "rollup_summary.json"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    # op set
    if args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    elif args.source:
        data = json.loads(Path(args.source).read_text(encoding="utf-8"))
        ops = sorted(data.keys()) if isinstance(data, dict) else sorted(data)
    else:
        ops = sorted(p.parent.parent.name
                     for p in runs_root.glob("*/operator/summary.json"))
    if not ops:
        print("no ops found (give --source / --ops, or ensure runs/*/operator/summary.json exist)")
        return

    perf = {}
    perf_path = Path(args.perf)
    if perf_path.exists():
        try:
            perf = json.loads(perf_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            perf = {}

    rows = [op_row(op, runs_root, perf, args.backend) for op in ops]
    summary = summarize(rows)

    # write CSV
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    # print summary
    def _fmt(d):
        return f"{d['pass']}/{d['of']} ({d['rate']*100:.1f}%)" if d.get("of") else "n/a"

    print("=" * 72)
    print(f"ROLL-UP  ({summary['n_operator_runs']}/{summary['n_ops']} ops have an operator run, "
          f"perf backend={args.backend})")
    print("-" * 72)
    print(f"  compile correctness      : {_fmt(summary['compile_correctness'])}")
    print(f"  kernel numeric (vs torch): {_fmt(summary['kernel_numeric'])}")
    print(f"  e2e numeric (ncnn::Net)  : {_fmt(summary['e2e_numeric'])}")
    print(f"  production correctness   : {_fmt(summary['production_correctness'])}")
    dg = summary["device_gate"]
    if dg["ran"]:
        print(f"  device gate (on phone)   : {dg['passed']}/{dg['ran']} "
              f"({dg['rate']*100:.1f}%)  [skipped/no-device: {dg['skipped']}]")
    else:
        print(f"  device gate (on phone)   : not run (device-verify off / no device)")
    si = summary["device_speedup_inline"]
    if si.get("n"):
        print(f"  device speedup (inline)  : n={si['n']} median={si['median']}x mean={si['mean']}x "
              f"faster_than_native={si['n_faster_than_native']}/{si['n']} (range {si['min']}–{si['max']}x)")
    print(f"  already in ncnn (native) : {_fmt(summary['already_in_ncnn'])}")
    for tier in ("speedup_shipped", "speedup_fair"):
        s = summary[tier]
        if s.get("n"):
            print(f"  {tier:<24} : n={s['n']} median={s['median']}x mean={s['mean']}x "
                  f"faster_than_native={s['n_faster_than_native']}/{s['n']} "
                  f"(range {s['min']}–{s['max']}x)")
        else:
            print(f"  {tier:<24} : no perf data (run run_perf_compare.py first)")
    print("=" * 72)
    print(f"per-op CSV -> {out_csv}")
    print(f"summary    -> {args.out_json}")


if __name__ == "__main__":
    main()
