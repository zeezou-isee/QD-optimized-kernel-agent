"""Diagnose why some kernels compile on host (Mac arm64) but FAIL the android
NDK cross-compile — the 'our-kernel benchncnn build failed' rows in the sweep.

For each op: install Cand_<Op> (base [+arm]) into the ncnn tree, run the android
`make benchncnn`, capture the compiler errors, restore the tree. Tells us whether
each failure is a per-kernel authoring bug (design/portability defect we can fix)
or a systematic NDK issue.

Usage:
    python scripts/diag_ndk_build.py --ops Erf,Log,AveragePool,Conv3D
    python scripts/diag_ndk_build.py --ops Erf --base-only   # isolate base vs arm
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/xingze/Documents/project/kernelgen/QD-optimized-kernel-agent")
NCNN = Path("/Users/xingze/Documents/project/kernelgen/ncnn")
BUILD = NCNN / "build-android-aarch64"
RUNS = REPO / "opgen" / "runs"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opgen"))
sys.path.insert(0, str(REPO / "opgen" / "orchestrator"))
sys.path.insert(0, str(REPO / "ncnn_kernel_test"))
import opgen; opgen.bootstrap_paths()
from layer_oracle import NetOracle


def base_arm_code(op: str, use_final: bool = True):
    import json
    art = RUNS / op / "base_kernel/artifacts"
    base = {p.name: p.read_text() for p in art.glob("*") if p.suffix in (".h", ".cpp")}
    arm = {}
    sj = RUNS / op / "backends/arm/kernel/summary.json"
    if use_final and sj.exists():
        # the VERIFIED winning code — NOT rounds[-1], which is a leftover failed attempt
        rc = (json.loads(sj.read_text()).get("final_result") or {}).get("response_code") or {}
        arm = {k: v for k, v in rc.items() if k.endswith(("_arm.h", "_arm.cpp"))}
    if not arm:
        rounds = sorted((RUNS / op / "backends/arm/kernel").glob("round_*"))
        if rounds:
            arm = {p.name: p.read_text() for p in rounds[-1].glob("*")
                   if p.name.endswith(("_arm.h", "_arm.cpp"))}
    return base, arm


def diag(op: str, base_only: bool) -> dict:
    base, arm = base_arm_code(op)
    if not base:
        return {"op": op, "result": "no base artifacts"}
    cls = f"Cand_{op}"
    netoc = NetOracle(ncnn_root=NCNN, workdir=RUNS / op / "_diag_net")
    handles = [netoc.install_layer(base, cls)]
    if arm and not base_only:
        handles.append(netoc.install_layer(arm, cls, subdir="arm", add_cmake=False))
    try:
        r = subprocess.run(["make", "-j", "8", "benchncnn"], cwd=BUILD,
                           capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
        errs = [l.strip() for l in out.splitlines() if "error:" in l.lower()]
        # which file(s) the errors are in
        files = sorted({l.split(":")[0].split("/")[-1] for l in errs if ".cpp" in l or ".h" in l})
        return {"op": op, "rc": r.returncode, "n_errors": len(errs),
                "files": files, "errors": errs[:6],
                "arm_installed": bool(arm and not base_only)}
    finally:
        for h in reversed(handles):
            netoc.restore(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--base-only", action="store_true",
                    help="install only base (isolate whether the base or arm kernel fails)")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    for op in ops:
        print(f"\n{'='*70}\n=== {op} (arm_installed={not args.base_only}) ===")
        try:
            d = diag(op, args.base_only)
        except Exception as exc:  # noqa: BLE001
            d = {"op": op, "result": f"crash: {exc}"}
        if d.get("result"):
            print(" ", d["result"]); continue
        print(f"  rc={d['rc']}  n_errors={d['n_errors']}  files={d['files']}")
        for e in d["errors"]:
            print("   ", e[:200])


if __name__ == "__main__":
    main()
