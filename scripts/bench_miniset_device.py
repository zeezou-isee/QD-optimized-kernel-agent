"""P0 device-in-the-loop measurement — batch benchmark of generated kernels on
the REAL phone (never Mac for final perf).

For each op: measure ncnn NATIVE, our BASE (portable C++), and our ARM (NEON)
on-device via op_profiler (benchncnn under simpleperf → clean loop timing + PMU).
Mechanism per backend: install Cand_<Op> into the ncnn tree → incremental
android rebuild → retarget the model's output layer to Cand_<Op> → profile on
device → restore the tree (no pollution).

Usage:
    python scripts/bench_miniset_device.py --ops Abs,Add        # subset
    python scripts/bench_miniset_device.py                       # all miniset
    python scripts/bench_miniset_device.py --backends native,arm # skip base
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/Users/xingze/Documents/project/kernelgen/QD-optimized-kernel-agent")
NCNN = Path("/Users/xingze/Documents/project/kernelgen/ncnn")
BUILD = NCNN / "build-android-aarch64"
DEVDIR = "/data/local/tmp/ncnn"
DATASET = REPO / "dataset" / "Mobilekernelbench_miniset"
OUT = REPO / "batch" / "results" / "miniset_device.json"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opgen"))
sys.path.insert(0, str(REPO / "opgen" / "orchestrator"))
sys.path.insert(0, str(REPO / "ncnn_kernel_test"))
import opgen; opgen.bootstrap_paths()
from layer_oracle import NetOracle, retarget_param_output_layer
from production_validation import torch_input_shapes_str
import op_profiler

MINISET = ["Abs", "Add", "And", "AveragePool", "BatchNormalization",
           "Conv", "Gemm", "Greater", "Mul", "ReduceMax", "ReduceSum"]


def adb(*a, **k):
    return subprocess.run(["adb", *a], capture_output=True, text=True, **k)


def find_param(op: str) -> Path | None:
    for sub in ("operator/_baseline_probe/_probe", "analyze/pnnx_probe/_probe",
                "graph/_probe", "graph/round_00"):
        p = REPO / "opgen/runs" / op / sub / f"{op}.ncnn.param"
        if p.exists():
            return p
    hits = sorted((REPO / "opgen/runs" / op).rglob(f"{op}.ncnn.param"))
    return hits[0] if hits else None


def find_model(op: str) -> Path | None:
    hits = sorted(DATASET.rglob(f"{op}.py"))
    return hits[0] if hits else None


def scale_shapes(shapes: str, factor: int) -> str:
    """Multiply the FIRST dim (w) of each input bracket by `factor` so the kernel
    dominates the net latency (miniset default shapes are too small → framework-
    bound). Scaling only the first dim keeps the last dim (c) intact, so per-
    channel weights (e.g. BatchNorm) stay compatible. Safe for single-input ops."""
    import re as _re
    def _one(m):
        nums = [x.strip() for x in m.group(1).split(",")]
        if nums:
            nums[0] = str(int(nums[0]) * factor)
        return "[" + ",".join(nums) + "]"
    return _re.sub(r"\[([^\]]+)\]", _one, shapes)


def native_type(param: Path) -> str:
    """The op's native ncnn layer type = the last non-plumbing layer's type."""
    skip = {"Input", "Output", "Split"}
    last = ""
    for ln in param.read_text().splitlines()[2:]:
        parts = ln.split()
        if parts and parts[0] not in skip:
            last = parts[0]
    return last


def base_arm_code(op: str) -> tuple[dict, dict]:
    art = REPO / "opgen/runs" / op / "base_kernel/artifacts"
    base = {p.name: p.read_text() for p in art.glob("*") if p.suffix in (".h", ".cpp")}
    # VERIFIED arm code from summary.final_result — NOT rounds[-1] (which holds
    # leftover FAILED attempts: wrong member names, hallucinated intrinsics, etc).
    arm = {}
    sj = REPO / "opgen/runs" / op / "backends/arm/kernel/summary.json"
    if sj.exists():
        rc = (json.loads(sj.read_text()).get("final_result") or {}).get("response_code") or {}
        arm = {k: v for k, v in rc.items() if k.endswith(("_arm.h", "_arm.cpp"))}
    if not arm:
        rounds = sorted((REPO / "opgen/runs" / op / "backends/arm/kernel").glob("round_*"))
        if rounds:
            arm = {p.name: p.read_text() for p in rounds[-1].glob("*")
                   if p.name.endswith(("_arm.h", "_arm.cpp"))}
    return base, arm


def kernel_class_name(op: str) -> str:
    """Real class name from kernel_profile.json (LLM may shorten it), not f'Cand_{op}'."""
    p = REPO / "opgen/runs" / op / "base_kernel/artifacts/kernel_profile.json"
    if p.exists():
        cn = (json.loads(p.read_text()) or {}).get("class_name")
        if cn:
            return cn
    return f"Cand_{op}"


def android_rebuild() -> bool:
    r = subprocess.run(["cmake", "--build", ".", "-j", "8", "--target", "benchncnn"],
                       cwd=BUILD, capture_output=True, text=True, timeout=1800)
    ok = r.returncode == 0 and (BUILD / "benchmark" / "benchncnn").exists()
    if not ok:
        print("    [build FAIL]", r.stderr[-500:])
    return ok


def push_benchncnn():
    adb("push", str(BUILD / "benchmark" / "benchncnn"), f"{DEVDIR}/benchncnn", timeout=60)
    adb("shell", "chmod", "+x", f"{DEVDIR}/benchncnn", timeout=10)


def profile(op: str, shapes: str) -> dict:
    r = op_profiler.profile_operator(op, "model.param", shapes, threads=1,
                                     loop=20000, device_dir=DEVDIR, simpleperf_cmd="simpleperf")
    return {"latency_min": r.get("latency_min"), "latency_avg": r.get("latency_avg"),
            "ipc": r.get("ipc"), "symbol": r.get("operator_symbol"),
            "trustworthy": r.get("trustworthy"), "error": r.get("error")}


def bench_op(op: str, backends: list[str], scale: int = 1) -> dict:
    param = find_param(op); model = find_model(op)
    if not param or not model:
        return {"op": op, "error": f"missing param={param} model={model}"}
    shapes = torch_input_shapes_str(model)
    if scale > 1:
        shapes = scale_shapes(shapes, scale)
    nt = native_type(param)
    cls = kernel_class_name(op)
    res = {"op": op, "native_type": nt, "shapes": shapes, "results": {}}
    print(f"\n=== {op} (native_type={nt}, shapes={shapes}) ===")

    # 1. NATIVE (no install, original param) — current benchncnn runs built-in
    if "native" in backends:
        adb("push", str(param), f"{DEVDIR}/model.param", timeout=30)
        res["results"]["native"] = profile(op, shapes)
        print(f"  native: {res['results']['native']['latency_min']} ms  {res['results']['native']['symbol']}")

    base_code, arm_code = base_arm_code(op)
    netoc = NetOracle(ncnn_root=NCNN, workdir=REPO / "opgen/runs" / op / "_bench_net")

    # 2. our BASE (install base only -> Cand_<Op> uses portable C++ forward)
    if "base" in backends and base_code:
        h = netoc.install_layer(base_code, cls)
        try:
            if android_rebuild():
                push_benchncnn()
                nt_txt, n = retarget_param_output_layer(param.read_text(), cls, expected_src_type=nt)
                if n:
                    rp = REPO / "opgen/runs" / op / "_bench_net" / "base.param"
                    rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(nt_txt)
                    adb("push", str(rp), f"{DEVDIR}/model.param", timeout=30)
                    res["results"]["our_base"] = profile(op, shapes)
                    print(f"  our_base: {res['results']['our_base']['latency_min']} ms  {res['results']['our_base']['symbol']}")
                else:
                    res["results"]["our_base"] = {"error": f"retarget matched 0 (nt={nt})"}
        finally:
            netoc.restore(h)

    # 3. our ARM (install base+arm -> Cand_<Op>_arm auto-selected on device arm64)
    if "arm" in backends and base_code and arm_code:
        hb = netoc.install_layer(base_code, cls)
        ha = netoc.install_layer(arm_code, cls, subdir="arm", add_cmake=False)
        try:
            if android_rebuild():
                push_benchncnn()
                nt_txt, n = retarget_param_output_layer(param.read_text(), cls, expected_src_type=nt)
                if n:
                    rp = REPO / "opgen/runs" / op / "_bench_net" / "arm.param"
                    rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(nt_txt)
                    adb("push", str(rp), f"{DEVDIR}/model.param", timeout=30)
                    res["results"]["our_arm"] = profile(op, shapes)
                    print(f"  our_arm: {res['results']['our_arm']['latency_min']} ms  {res['results']['our_arm']['symbol']}")
                else:
                    res["results"]["our_arm"] = {"error": f"retarget matched 0 (nt={nt})"}
        finally:
            netoc.restore(ha); netoc.restore(hb)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=None, help="comma list; default all miniset")
    ap.add_argument("--backends", default="native,base,arm")
    ap.add_argument("--scale", type=int, default=1,
                    help="multiply each input's first dim by this (make the kernel "
                         "dominate net latency; safe for single-input ops)")
    ap.add_argument("--out", default=None, help="override results JSON path")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",")] if args.ops else MINISET
    backends = [b.strip() for b in args.backends.split(",")]
    global OUT
    if args.out:
        OUT = Path(args.out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    for op in ops:
        try:
            r = bench_op(op, backends, scale=args.scale)
        except Exception as exc:  # noqa: BLE001
            r = {"op": op, "error": f"crashed: {exc}"}
            print(f"  [CRASH] {op}: {exc}")
        allres[op] = r
        OUT.write_text(json.dumps(allres, ensure_ascii=False, indent=2))

    # summary table
    print("\n" + "=" * 74)
    print(f"{'op':<20}{'native':>10}{'our_base':>12}{'our_arm':>10}{'arm/native':>12}")
    print("-" * 74)
    for op in ops:
        r = allres.get(op, {}); rr = r.get("results", {})
        def g(k):
            v = rr.get(k, {}); return v.get("latency_min") if isinstance(v, dict) else None
        nat, ba, ar = g("native"), g("our_base"), g("our_arm")
        ratio = f"{ar/nat:.2f}x" if (nat and ar) else "-"
        print(f"{op:<20}{str(nat):>10}{str(ba):>12}{str(ar):>10}{ratio:>12}")
    print("=" * 74)
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
