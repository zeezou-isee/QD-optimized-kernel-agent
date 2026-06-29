"""Measure the on-device (Android) performance of NATIVE ncnn operators.

For models already converted by convert_dataset.py (dataset/converted/<cat>/<Op>/
<Op>.ncnn.param), this measures the performance of ncnn's OWN built-in kernel —
no custom kernel is generated, retarget_to is left None — so the numbers are a
NATIVE BASELINE to compare future optimized kernels against.

One on-device measurement per operator (reuses ProductionValidator.profile_op,
no reimplementation): benchncnn is run under simpleperf for each thread config
(1 & 2), and each config yields BOTH micro-architecture PMU metrics (IPC /
cache-miss / branch-miss / operator fraction) AND latency (latency_avg/min/max).
A single run produces both — there is no separate benchmark pass.

Input shape is inferred from the original PyTorch reference model under
dataset/Mobilekernelbench/<cat>/<Op>.py (torch_input_shapes_str).

Requirements: a connected android device (adb), ANDROID_NDK set (to cross-compile
benchncnn). simpleperf is taken from the device if present, else pushed from NDK.
Missing prerequisites -> that op is SKIPPED (not failed), same as the pipeline.

    python profile_native_baseline.py --only Softmax
    python profile_native_baseline.py --category Activation
    python profile_native_baseline.py                      # all converted ops
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))            # import opgen
sys.path.insert(0, str(ROOT / "opgen"))  # flat imports (config, production_validation)
import opgen as _opgen; _opgen.bootstrap_paths()

from config import GraphConfig                                    # noqa: E402
from production_validation import (                               # noqa: E402
    ProductionValidator,
    torch_input_shapes_str,
)


def discover(converted_root: Path, src_root: Path,
             only: set[str] | None, category: str | None) -> list[dict]:
    """Pair each converted <Op>.ncnn.param with its source .py (for shape)."""
    rows = []
    for param in sorted(converted_root.rglob("*.ncnn.param")):
        op = param.stem.replace(".ncnn", "")  # <Op>.ncnn.param -> <Op>
        cat = param.parent.parent.name
        if only and op not in only:
            continue
        if category and cat != category:
            continue
        src = src_root / cat / f"{op}.py"
        rows.append({"op": op, "category": cat, "param": param,
                     "src": src if src.exists() else None})
    return rows


def measure_one(pv: ProductionValidator, row: dict) -> dict:
    op, param, src = row["op"], row["param"], row["src"]
    out = {"op": op, "category": row["category"], "param": str(param.relative_to(ROOT))}
    if src is None:
        out["error"] = f"source model not found for shape inference: {op}.py"
        return out
    try:
        shape = torch_input_shapes_str(src)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"shape inference failed: {exc}"
        return out
    out["shape"] = shape

    t0 = time.time()
    # NATIVE baseline: retarget_to=None -> measure ncnn's built-in kernel as-is.
    # profile_op runs benchncnn under simpleperf, so each per-thread config carries
    # BOTH micro-arch metrics and latency_{avg,min,max} — one on-device run.
    out["profile"] = pv.profile_op(str(param), shape, op_name=op, retarget_to=None)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted", default="dataset/converted",
                    help="root of converted models (from convert_dataset.py)")
    ap.add_argument("--src", default="dataset/Mobilekernelbench",
                    help="root of original .py models (for input-shape inference)")
    ap.add_argument("--only", default=None, help="comma list of op names")
    ap.add_argument("--category", default=None, help="restrict to one category dir")
    ap.add_argument("--out", default="dataset/converted/native_baseline_perf.json")
    ap.add_argument("--ncnn-root", default=None)
    ap.add_argument("--profile-loop", type=int, default=10000,
                    help="benchncnn loop_count under simpleperf (default 10000); "
                         "this run also yields the latency numbers")
    args = ap.parse_args()

    converted_root = (ROOT / args.converted) if not Path(args.converted).is_absolute() else Path(args.converted)
    src_root = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    if not converted_root.is_dir():
        sys.exit(f"converted dir not found: {converted_root}\nRun convert_dataset.py first.")

    cfg_kwargs = {"run_numeric": False}
    if args.ncnn_root:
        cfg_kwargs["ncnn_root"] = args.ncnn_root
    ncnn_root = GraphConfig(**cfg_kwargs).ncnn_root

    pv = ProductionValidator(ncnn_root=ncnn_root, compile_mode="build_lib",
                             do_benchmark=True, workdir=ROOT / "opgen" / "runs" / "_native_perf",
                             profile_loop=args.profile_loop)

    rows = discover(converted_root, src_root, only, args.category)
    print(f"[perf] {len(rows)} converted ops; measuring NATIVE ncnn baseline")
    print(f"[perf] ncnn_root={ncnn_root}\n")

    results = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['category']}/{row['op']} ...", flush=True)
        r = measure_one(pv, row)
        results.append(r)
        pf = r.get("profile", {})
        c0 = (pf.get("configs") or [{}])[0]
        if r.get("error"):
            print(f"        ERROR {r['error']}", flush=True)
        else:
            print(f"        latency_avg={c0.get('latency_avg')}ms | "
                  f"ipc={c0.get('ipc')} frac={c0.get('operator_fraction')} "
                  f"(profile ran={pf.get('ran')} reason={pf.get('reason','')}) "
                  f"{r.get('elapsed_s')}s", flush=True)

    n_prof = sum(1 for r in results if r.get("profile", {}).get("ran"))
    summary = {"total": len(results), "profiled": n_prof, "results": results}
    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[perf] DONE: profiled={n_prof} / {len(results)} (each carries "
          f"latency + micro-arch per thread). -> {out_path}")


if __name__ == "__main__":
    main()
