"""CLI — perf comparison: OUR generated kernel vs ncnn's NATIVE built-in op.

Standalone, self-contained tool (does not touch the OperatorAgent flow). For a
given op + backend it:

  1. benchmarks OUR generated kernel on the REAL phone (never Mac for final perf)
     via the proven install -> android rebuild -> retarget-output-layer ->
     benchncnn-under-simpleperf -> restore path;
  2. IF `--perf-comp-base` is set AND ncnn NATIVELY supports the op (reusing the
     existence check already computed by the OperatorAgent, with a layer-interface
     fallback), ALSO benchmarks the ncnn built-in op on the SAME backend and
     computes the speedup ratio (native_latency / ours_latency; >1 = ours faster).

Precision fairness (see FINAL_miniset_results.md): benchncnn hardcodes
fp16+packing=true, so a naive native-vs-ours ratio compares ncnn's fp16+packed
path against our fp32 kernel. We therefore record BOTH tiers per op:
  - "shipped": stock benchncnn (native fp16+packing, ours fp32 wrapped in packing);
  - "fair"   : fp16=0 packing=0 (both fp32, no packing wrap) — needs the opt-in
    `fp16=`/`packing=` args added to benchncnn.cpp.

Results are merged incrementally into batch/results/perf_compare.json, keyed by
"<op>:<backend>", for later statistics.

Usage:
    python opgen/cli/run_perf_compare.py --task Abs --backend arm --perf-comp-base
    python opgen/cli/run_perf_compare.py --ops Abs,BatchNormalization,Conv \
        --backend arm --perf-comp-base --scale 8
    python opgen/cli/run_perf_compare.py --task Conv --backend vulkan --perf-comp-base
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

# bootstrap opgen flat-import paths (we live in opgen/cli/)
REPO = Path(__file__).resolve().parents[2]
NCNN = Path("/Users/xingze/Documents/project/kernelgen/ncnn")
BUILD = NCNN / "build-android-aarch64"
BUILD_VK = NCNN / "build-android-vk"
DEVDIR = "/data/local/tmp/ncnn"
# Full set (miniset ops are a subset; find_model uses rglob so both resolve).
# Override with --dataset for a different corpus.
DATASET = REPO / "dataset" / "Mobilekernelbench"
RUNS = REPO / "opgen" / "runs"
OUT = REPO / "batch" / "results" / "perf_compare.json"
IFACE = REPO / "experience_pool" / "backend_ncnn" / "layer_interfaces.json"

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


# ---------------------------------------------------------------- small helpers
def adb(*a, **k):
    return subprocess.run(["adb", *a], capture_output=True, text=True, **k)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def find_param(op: str) -> Path | None:
    """The pnnx-emitted baseline .ncnn.param (native conversion, before our custom
    layer). Prefer the operator agent's baseline probe, then the shared analyze
    probe, then any hit."""
    for sub in ("operator/_baseline_probe/_probe", "analyze/pnnx_probe/_probe",
                "graph/_probe", "graph/round_00"):
        p = RUNS / op / sub / f"{op}.ncnn.param"
        if p.exists():
            return p
    hits = sorted((RUNS / op).rglob(f"{op}.ncnn.param"))
    return hits[0] if hits else None


def find_model(op: str) -> Path | None:
    hits = sorted(DATASET.rglob(f"{op}.py"))
    return hits[0] if hits else None


def native_type(param: Path) -> str:
    """The op's native ncnn layer type = the last non-plumbing layer's type."""
    skip = {"Input", "Output", "Split"}
    last = ""
    for ln in param.read_text().splitlines()[2:]:
        parts = ln.split()
        if parts and parts[0] not in skip:
            last = parts[0]
    return last


def scale_shapes(shapes: str, factor: int) -> str:
    """Grow the input so the kernel dominates net latency (miniset defaults are
    framework-bound). Scales the first dim of ONLY the FIRST bracket (the main
    activation), leaving the rest untouched — weighted ops (Conv/Deconv/Gemm)
    carry their weight/bias as extra input blobs, and scaling those would corrupt
    the layer. The activation's channel dim (last) is kept intact too."""
    def _one(m):
        nums = [x.strip() for x in m.group(1).split(",")]
        if nums:
            nums[0] = str(int(nums[0]) * factor)
        return "[" + ",".join(nums) + "]"
    return re.sub(r"\[([^\]]+)\]", _one, shapes, count=1)


def base_arm_code(op: str) -> tuple[dict, dict]:
    art = RUNS / op / "base_kernel/artifacts"
    base = {p.name: p.read_text() for p in art.glob("*") if p.suffix in (".h", ".cpp")}
    arm = {}
    rounds = sorted((RUNS / op / "backends/arm/kernel").glob("round_*"))
    if rounds:
        arm = {p.name: p.read_text() for p in rounds[-1].glob("*")
               if p.name.endswith(("_arm.h", "_arm.cpp"))}
    return base, arm


_IFACE_NAMES: set[str] | None = None


def _iface_names() -> set[str]:
    global _IFACE_NAMES
    if _IFACE_NAMES is None:
        try:
            _IFACE_NAMES = {d["name"] for d in json.loads(IFACE.read_text())}
        except Exception:  # noqa: BLE001
            _IFACE_NAMES = set()
    return _IFACE_NAMES


def native_supported(op: str, nt: str) -> tuple[bool, str]:
    """Reuse the OperatorAgent's existence check if available; else fall back to
    'is the pnnx-emitted native layer type a real ncnn built-in'."""
    s = RUNS / op / "operator" / "summary.json"
    if s.exists():
        try:
            ec = (json.loads(s.read_text()).get("phases", {})
                  .get("existence_check", {}))
            if "already_in_ncnn" in ec:
                return bool(ec["already_in_ncnn"]), "operator.existence_check"
        except Exception:  # noqa: BLE001
            pass
    if nt and nt in _iface_names():
        return True, f"layer_interface({nt})"
    return False, f"native_type={nt or '∅'} not a ncnn built-in"


# ---------------------------------------------------------------- device build
def android_rebuild(build_dir: Path = BUILD, target: str = "benchncnn") -> bool:
    # The build dir uses the "Unix Makefiles" generator, so `make <target>` is the
    # direct equivalent and does not need cmake on PATH (which it often isn't).
    # Fall back to `cmake --build` for Ninja/other generators.
    for cmd in (["make", "-j", "8", target],
                ["cmake", "--build", ".", "-j", "8", "--target", target]):
        try:
            r = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError:
            continue
        if r.returncode == 0 and (build_dir / "benchmark" / target).exists():
            return True
        print("    [build FAIL]", (r.stderr or r.stdout)[-600:])
        return False
    print("    [build FAIL] neither make nor cmake available")
    return False


def push_benchncnn(build_dir: Path = BUILD):
    adb("push", str(build_dir / "benchmark" / "benchncnn"), f"{DEVDIR}/benchncnn", timeout=60)
    adb("shell", "chmod", "+x", f"{DEVDIR}/benchncnn", timeout=10)


def build_clean_benchncnn_once() -> bool:
    """Build the clean/native benchncnn ONCE per sweep and push it. Every op's
    native measurement reuses it (see bench_cpu step 1) instead of rebuilding the
    identical clean binary per op — the single biggest per-op cost is relinking
    the 63M libncnn.a + 35M benchncnn, and the clean binary is the same for all
    ops. The per-op 'ours' rebuild (with that op's Cand_) is still needed."""
    print("=== building clean benchncnn ONCE (shared native baseline) ===")
    if not android_rebuild():
        print("  [warn] clean benchncnn build failed — native measurements may be skipped")
        return False
    push_benchncnn()
    return True


# ---------------------------------------------------------------- CPU profiling
def _profile_cpu(op: str, shapes: str, loop: int, fair: bool, tier: str,
                 timeout: int = 600) -> dict:
    extra = ("fp16=0", "packing=0") if fair else ()
    r = op_profiler.profile_operator(op, "model.param", shapes, threads=1, loop=loop,
                                     device_dir=DEVDIR, simpleperf_cmd="simpleperf",
                                     bench_extra=extra, timeout=timeout)
    return {"tier": tier, "latency_min": r.get("latency_min"),
            "latency_avg": r.get("latency_avg"), "ipc": r.get("ipc"),
            "symbol": r.get("operator_symbol"), "trustworthy": r.get("trustworthy"),
            "error": r.get("error")}


def bench_cpu(op: str, backend: str, loop: int, scale: int, comp_base: bool,
              timeout: int = 600) -> dict:
    param = find_param(op); model = find_model(op)
    if not param or not model:
        return {"op": op, "backend": backend, "error": f"missing param={param} model={model}"}
    shapes = torch_input_shapes_str(model)
    if scale > 1:
        shapes = scale_shapes(shapes, scale)
    nt = native_type(param)
    cls = f"Cand_{op}"
    supported, why = native_supported(op, nt)
    res = {"op": op, "backend": backend, "native_type": nt, "shapes": shapes,
           "scale": scale, "runner": "benchncnn", "native_supported": supported,
           "native_support_src": why, "ts": _now()}
    print(f"\n=== {op} [{backend}] native_type={nt} supported={supported} ({why}) "
          f"shapes={shapes} ===")

    # 1) NATIVE first — reuses the clean benchncnn built ONCE at sweep start
    #    (build_clean_benchncnn_once). ncnn builtins are identical across ops, and
    #    the native run uses the ORIGINAL param which never references our Cand_
    #    layer, so a benchncnn that still has a prior op's Cand_ linked is harmless
    #    here. This avoids one full 63M-archive relink per op vs rebuilding clean.
    if comp_base and supported:
        adb("push", str(param), f"{DEVDIR}/model.param", timeout=30)
        res["native_shipped"] = _profile_cpu(op, shapes, loop, fair=False, tier="fp16+packing", timeout=timeout)
        print(f"  native_shipped: {res['native_shipped']['latency_min']} ms")
        res["native_fair"] = _profile_cpu(op, shapes, loop, fair=True, tier="fp32", timeout=timeout)
        print(f"  native_fair:    {res['native_fair']['latency_min']} ms")

    # 2) OURS — install Cand_<Op> (+arm) -> rebuild -> retarget -> profile -> restore.
    base_code, arm_code = base_arm_code(op)
    if not base_code:
        return {**res, "error": "no base_kernel artifacts for our kernel"}
    if backend == "arm" and not arm_code:
        return {**res, "error": "backend=arm but no arm kernel authored"}
    netoc = NetOracle(ncnn_root=NCNN, workdir=RUNS / op / "_perfcmp_net")
    handles = [netoc.install_layer(base_code, cls)]
    if backend == "arm":
        handles.append(netoc.install_layer(arm_code, cls, subdir="arm", add_cmake=False))
    try:
        if not android_rebuild():
            return {**res, "error": "our-kernel benchncnn build failed"}
        push_benchncnn()
        nt_txt, n = retarget_param_output_layer(param.read_text(), cls, expected_src_type=nt)
        if not n:
            return {**res, "error": f"retarget matched 0 (nt={nt})"}
        rp = RUNS / op / "_perfcmp_net" / "ours.param"
        rp.parent.mkdir(parents=True, exist_ok=True); rp.write_text(nt_txt)
        adb("push", str(rp), f"{DEVDIR}/model.param", timeout=30)
        res["ours_shipped"] = _profile_cpu(op, shapes, loop, fair=False, tier="fp32(shipped-cfg)", timeout=timeout)
        print(f"  ours_shipped:   {res['ours_shipped']['latency_min']} ms")
        if comp_base and supported:
            res["ours_fair"] = _profile_cpu(op, shapes, loop, fair=True, tier="fp32", timeout=timeout)
            print(f"  ours_fair:      {res['ours_fair']['latency_min']} ms")
    finally:
        for h in reversed(handles):
            netoc.restore(h)

    _attach_speedups(res)
    return res


# ---------------------------------------------------------------- Vulkan (GPU)
def _benchncnn_vk_latency(param: Path, shapes: str, loop: int, fair: bool) -> dict:
    """Native GPU op via benchncnn gpu=0 on the ORIGINAL param (ncnn built-in
    vulkan layer). Different runner from our oracle-runner path (flagged
    cross_runner in the result) but same GPU/backend."""
    extra = " fp16=0 packing=0" if fair else ""
    adb("push", str(param), f"{DEVDIR}/vk_model.param", timeout=30)
    cmd = (f"cd {DEVDIR} && ./benchncnn {loop} 1 2 0 0 "
           f"param=vk_model.param shape='{shapes}'{extra} 2>&1")
    out = adb("shell", cmd, timeout=180).stdout
    lat = op_profiler._parse_latency(out) if hasattr(op_profiler, "_parse_latency") else {}
    return {"tier": "fp32" if fair else "fp16+packing",
            "runner": "benchncnn-vk", "latency_min": lat.get("latency_min"),
            "latency_avg": lat.get("latency_avg"),
            "error": None if lat.get("latency_min") is not None else out.strip()[-200:]}


def bench_vulkan(op: str, loop: int, comp_base: bool) -> dict:
    param = find_param(op)
    nt = native_type(param) if param else ""
    supported, why = native_supported(op, nt)
    res = {"op": op, "backend": "vulkan", "native_type": nt, "native_supported": supported,
           "native_support_src": why, "cross_runner": True, "ts": _now(),
           "note": "vulkan: ours=oracle-runner single-op GPU dispatch; "
                   "native=benchncnn gpu=0 whole-net — cross-runner approximation"}
    print(f"\n=== {op} [vulkan] native_type={nt} supported={supported} ===")

    # OURS: reuse the proven from-scratch-vulkan device path (oracle runner + --bench).
    subprocess.run([sys.executable, str(REPO / "scripts" / "bench_vulkan_device.py"),
                    "--ops", op], capture_output=True, text=True, timeout=1800)
    ours = {}
    vk_out = REPO / "batch/results/miniset_vulkan_device.json"
    if vk_out.exists():
        ours = json.loads(vk_out.read_text()).get(op, {})
    res["ours"] = {"gpu_latency_min_ms": ours.get("gpu_latency_min_ms"),
                   "max_diff_vs_host": ours.get("max_diff_vs_host"),
                   "ran": ours.get("ran"), "runner": "oracle",
                   "error": ours.get("error")}
    print(f"  ours(oracle): {res['ours']['gpu_latency_min_ms']} ms  err={res['ours'].get('error','')}")

    # NATIVE: benchncnn gpu=0 (needs a vulkan benchncnn build).
    if comp_base and supported and param:
        vk_bin = BUILD_VK / "benchmark" / "benchncnn"
        if not vk_bin.exists() and not android_rebuild(BUILD_VK):
            res["native_shipped"] = {"error": "no vulkan benchncnn build (build-android-vk)"}
        else:
            push_benchncnn(BUILD_VK)
            model = find_model(op)
            shapes = torch_input_shapes_str(model) if model else "[1,32,64,64]"
            res["native_shipped"] = _benchncnn_vk_latency(param, shapes, loop, fair=False)
            res["native_fair"] = _benchncnn_vk_latency(param, shapes, loop, fair=True)
            print(f"  native_shipped: {res['native_shipped'].get('latency_min')} ms  "
                  f"native_fair: {res['native_fair'].get('latency_min')} ms")

    # speedup on GPU ms (ours oracle single-op vs native whole-net; approximate)
    ours_ms = res["ours"].get("gpu_latency_min_ms")
    for tier in ("shipped", "fair"):
        nat = (res.get(f"native_{tier}") or {}).get("latency_min")
        if isinstance(nat, (int, float)) and isinstance(ours_ms, (int, float)) and ours_ms:
            res[f"speedup_{tier}"] = round(nat / ours_ms, 3)
    return res


# ---------------------------------------------------------------- speedup calc
def _attach_speedups(res: dict) -> None:
    def _lm(k):
        v = res.get(k) or {}
        return v.get("latency_min")
    ns, os_ = _lm("native_shipped"), _lm("ours_shipped")
    nf, of = _lm("native_fair"), _lm("ours_fair")
    if isinstance(ns, (int, float)) and isinstance(os_, (int, float)) and os_:
        res["speedup_shipped"] = round(ns / os_, 3)
    if isinstance(nf, (int, float)) and isinstance(of, (int, float)) and of:
        res["speedup_fair"] = round(nf / of, 3)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Perf compare: our generated kernel vs ncnn native.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--task", default=None, help="single op")
    g.add_argument("--ops", default=None, help="comma list (default: miniset)")
    ap.add_argument("--backend", choices=["base", "arm", "vulkan"], default="arm",
                    help="which of OUR kernels to measure; native is measured on the "
                         "same compute backend (CPU for base/arm, GPU for vulkan).")
    ap.add_argument("--perf-comp-base", action="store_true",
                    help="when ncnn NATIVELY supports the op, also benchmark the native "
                         "op on the same backend and compute the speedup ratio.")
    ap.add_argument("--scale", type=int, default=1,
                    help="multiply each input's first dim (make the kernel dominate).")
    ap.add_argument("--loop", type=int, default=20000, help="benchncnn loop count")
    ap.add_argument("--record-timeout", type=int, default=600,
                    help="simpleperf record timeout (s); raise for slow fp16 paths on big shapes")
    ap.add_argument("--out", default=None, help="override results JSON path")
    ap.add_argument("--dataset", default=None,
                    help="override the dataset root used to resolve <op>.py input shapes "
                         "(default: full Mobilekernelbench)")
    args = ap.parse_args()

    global DATASET
    if args.dataset:
        DATASET = Path(args.dataset)

    if args.task:
        ops = [args.task]
    elif args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    else:
        ops = MINISET

    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}

    # One-time clean benchncnn for the shared native baseline (CPU backends only;
    # vulkan builds its own). Skipped when not comparing against native.
    if args.backend in ("base", "arm") and args.perf_comp_base:
        build_clean_benchncnn_once()

    for op in ops:
        key = f"{op}:{args.backend}"
        try:
            if args.backend == "vulkan":
                r = bench_vulkan(op, args.loop, args.perf_comp_base)
            else:
                r = bench_cpu(op, args.backend, args.loop, args.scale, args.perf_comp_base,
                              timeout=args.record_timeout)
        except Exception as exc:  # noqa: BLE001
            r = {"op": op, "backend": args.backend, "error": f"crashed: {exc}", "ts": _now()}
            print(f"  [CRASH] {op}: {exc}")
        allres[key] = r
        OUT.write_text(json.dumps(allres, ensure_ascii=False, indent=2))

    # summary table
    print("\n" + "=" * 88)
    print(f"{'op':<20}{'backend':>8}{'ours':>10}{'nat(ship)':>11}{'nat(fair)':>11}"
          f"{'x_ship':>9}{'x_fair':>9}")
    print("-" * 88)
    for op in ops:
        r = allres.get(f"{op}:{args.backend}", {})
        if args.backend == "vulkan":
            ours = (r.get("ours") or {}).get("gpu_latency_min_ms")
        else:
            ours = (r.get("ours_shipped") or {}).get("latency_min")
        ns = (r.get("native_shipped") or {}).get("latency_min")
        nf = (r.get("native_fair") or {}).get("latency_min")
        xs = r.get("speedup_shipped", "-"); xf = r.get("speedup_fair", "-")
        print(f"{op:<20}{args.backend:>8}{str(ours):>10}{str(ns):>11}{str(nf):>11}"
              f"{str(xs):>9}{str(xf):>9}")
    print("=" * 88)
    print("speedup = native_latency / ours_latency  (>1 = OUR kernel is faster)")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
