"""Progress view for the full-dataset VULKAN device-in-the-loop run.

Reads batch/results/all_kernel_vulkan.json (batch results, one key per op) plus each
op's runs/<op>/backends/vulkan/kernel/summary.json for the device gate + inline
speedup. Prints: how many done, host-success, on-Adreno device-passed, how many got
an inline GPU speedup, and the speedup distribution + top wins.

Usage:  python scripts/vulkan_progress.py   [--total 185]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "batch" / "results" / "all_kernel_vulkan.json"
RUNS = REPO / "opgen" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=185)
    args = ap.parse_args()

    alive = subprocess.run(["pgrep", "-f", "batch_runner.py --set all --kernel-only"],
                           capture_output=True).returncode == 0
    print(f"process: {'RUNNING' if alive else 'not running (done or stopped)'}")

    if not RESULTS.exists():
        print(f"no results yet at {RESULTS}")
        return
    d = json.loads(RESULTS.read_text())
    done = len(d)
    host_ok = sum(1 for r in d.values() if r.get("status") == "success")

    dpass = dskip = dfail = 0
    spd: list[tuple[str, float]] = []
    for op in d:
        p = RUNS / op / "backends/vulkan/kernel/summary.json"
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        ds = s.get("device_status")
        dpass += (ds == "passed"); dskip += (ds == "skipped"); dfail += (ds == "failed")
        sp = s.get("device_speedup")
        if isinstance(sp, (int, float)):
            spd.append((op, sp))

    pct = 100 * done / args.total if args.total else 0
    last = list(d)[-1] if d else "-"
    print(f"progress : {done}/{args.total} ({pct:.0f}%)   last={last}")
    print(f"host     : {host_ok} success (vulkan authored + host MoltenVK verified)")
    print(f"device   : passed={dpass}  failed={dfail}  skipped={dskip}  (on Adreno)")
    if spd:
        xs = [x for _, x in spd]
        wins = sum(1 for x in xs if x > 1)
        print(f"speedup  : n={len(xs)}  median={st.median(xs):.2f}x  mean={st.mean(xs):.2f}x  "
              f"wins(>1)={wins}/{len(xs)}  range {min(xs):.2f}-{max(xs):.2f}")
        top = sorted(spd, key=lambda t: -t[1])[:6]
        print("  top: " + ", ".join(f"{op} {x:.1f}x" for op, x in top))
    else:
        print("speedup  : none measured yet")


if __name__ == "__main__":
    main()
