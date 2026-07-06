"""Quick progress view for the real-device perf sweep (run_perf_compare.py).

Reads batch/results/perf_compare.json (written incrementally, one op at a time)
and prints: how many of the target ops are done, the last op recorded, any
NDK-build failures, and running speedup stats (fair fp32 + shipped fp16 tiers).

Usage:
    python scripts/sweep_progress.py
    python scripts/sweep_progress.py --total 173   # override the expected count
    python scripts/sweep_progress.py --perf batch/results/perf_compare.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", default=str(REPO / "batch" / "results" / "perf_compare.json"))
    ap.add_argument("--total", type=int, default=173, help="expected op count")
    args = ap.parse_args()

    # is the sweep process still running?
    alive = subprocess.run(["pgrep", "-f", "run_perf_compare.py"],
                           capture_output=True).returncode == 0
    print(f"process: {'RUNNING' if alive else 'not running (done or stopped)'}")

    p = Path(args.perf)
    if not p.exists():
        print(f"no results yet at {p}")
        return
    d = json.loads(p.read_text())
    done = len(d)
    order = list(d)
    fails = [k for k, v in d.items() if v.get("error")]
    fair = [v["speedup_fair"] for v in d.values() if isinstance(v.get("speedup_fair"), (int, float))]
    ship = [v["speedup_shipped"] for v in d.values() if isinstance(v.get("speedup_shipped"), (int, float))]

    pct = 100 * done / args.total if args.total else 0
    print(f"progress : {done}/{args.total} ({pct:.0f}%)   last={order[-1] if order else '-'}")
    print(f"build-fail: {len(fails)}  {[k.split(':')[0] for k in fails]}")

    def _line(tag, xs):
        if not xs:
            print(f"  {tag}: (no ratios yet)")
            return
        wins = sum(1 for x in xs if x > 1.0)
        print(f"  {tag}: median={st.median(xs):.2f}x  mean={st.mean(xs):.2f}x  "
              f"wins(>1)={wins}/{len(xs)}  range {min(xs):.2f}-{max(xs):.2f}")

    print("speedup (native / ours; >1 = our kernel faster):")
    _line("fair fp32   ", fair)
    _line("shipped fp16", ship)


if __name__ == "__main__":
    main()
