"""P3 — end-to-end whole-network bench on device (Redmi SD778G, Adreno 642L).

Chains N miniset ops into a single ncnn .param graph (Abs -> BatchNorm -> Abs ->
ReduceSum on shape [1,32,32,32]), installs our Cand_<Op> + Cand_<Op>_arm
kernels for each hop, and measures end-to-end latency on-device in TWO
conditions:

  A) all-native  : original ncnn ops (UnaryOp/BatchNorm/Reduction) at the
                   fp16+packed baseline tier.
  B) ours-swapped: every hop rewritten to Cand_<Op> — our fp32 kernels compiled
                   into the ncnn tree, no fp16, no packing.

Same graph, same shapes, same runner, same phone. Reports both totals + a
per-op contribution guess (delta between A and B on that hop, isolated by
partial swap runs).

This is a whole-network integration test — proves the per-op wins translate
to a real graph, or exposes where they don't (framework overhead per op,
tensor conversions, cache re-warm).
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
OUT = REPO / "batch" / "results" / "e2e_chain.json"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opgen"))
sys.path.insert(0, str(REPO / "opgen" / "orchestrator"))
sys.path.insert(0, str(REPO / "ncnn_kernel_test"))
import opgen; opgen.bootstrap_paths()
from layer_oracle import NetOracle
import op_profiler


def adb(*a, **k):
    return subprocess.run(["adb", *a], capture_output=True, text=True, **k)


def base_arm_code(op: str) -> tuple[dict, dict]:
    art = REPO / "opgen/runs" / op / "base_kernel/artifacts"
    base = {p.name: p.read_text() for p in art.glob("*") if p.suffix in (".h", ".cpp")}
    arm = {}
    rounds = sorted((REPO / "opgen/runs" / op / "backends/arm/kernel").glob("round_*"))
    if rounds:
        arm = {p.name: p.read_text() for p in rounds[-1].glob("*")
               if p.name.endswith(("_arm.h", "_arm.cpp"))}
    return base, arm


def android_rebuild() -> bool:
    # Use plain make (no cmake needed at run time — CMakeCache already exists).
    r = subprocess.run(["make", "-j", "8", "benchncnn"],
                       cwd=BUILD, capture_output=True, text=True, timeout=1800)
    ok = r.returncode == 0 and (BUILD / "benchmark" / "benchncnn").exists()
    if not ok:
        print("    [build FAIL]", (r.stderr or r.stdout)[-800:])
    return ok


def push_benchncnn():
    adb("push", str(BUILD / "benchmark" / "benchncnn"), f"{DEVDIR}/benchncnn", timeout=60)
    adb("shell", "chmod", "+x", f"{DEVDIR}/benchncnn", timeout=10)


# ---------------------------------------------------------------- graph builder
CHAIN_OPS = [
    ("Abs", "UnaryOp", "abs_0", "0=0"),
    ("BatchNormalization", "BatchNorm", "bn_0", "0=32 1=1.000000e-5"),
    ("Abs", "UnaryOp", "abs_1", "0=0"),
    ("ReduceSum", "Reduction", "sum_0", "0=0 1=0 -23303=1,1 4=0 5=1"),
]

INPUT_SHAPE = [1, 32, 256, 256]   # 8x8 spatial vs baseline — kernel dominates net overhead


def build_param(chain: list[tuple[str, str, str, str]]) -> str:
    """Emit a linear ncnn .param that pipes Input -> op1 -> op2 -> ... -> Output.
    Each element = (op_name_hint, ncnn_layer_type, layer_id, param_body)."""
    n_layers = 1 + len(chain)
    n_blobs = 1 + len(chain)
    lines = [
        "7767517",
        f"{n_layers} {n_blobs}",
        f"Input                    in0                      0 1 in0",
    ]
    prev = "in0"
    for i, (_hint, typ, lid, body) in enumerate(chain):
        out_blob = f"out{i}"
        lines.append(f"{typ:24} {lid:24} 1 1 {prev} {out_blob} {body}")
        prev = out_blob
    return "\n".join(lines) + "\n"


def build_bin_bytes(chain: list[tuple[str, str, str, str]], num_features: int = 32) -> bytes:
    """Only BatchNorm has weights in our chain — pack {gamma, mean, var, beta}
    all as ones/zeros so the forward is numerically inert (identity-ish)."""
    import numpy as np
    parts: list[np.ndarray] = []
    for _hint, typ, _lid, _body in chain:
        if typ == "BatchNorm":
            # ncnn BatchNorm order: slope(gamma), mean, var, bias(beta)
            parts.append(np.ones(num_features, dtype=np.float32))    # slope
            parts.append(np.zeros(num_features, dtype=np.float32))   # mean
            parts.append(np.ones(num_features, dtype=np.float32))    # var
            parts.append(np.zeros(num_features, dtype=np.float32))   # bias
    if not parts:
        return b""
    # Each weight block starts with a 4-byte fp32 tag header (raw = 0 = "raw fp32")
    tag = np.array([0], dtype=np.uint32).tobytes()
    return b"".join(tag + p.tobytes() for p in parts)


# ---------------------------------------------------------------- swap variants
def swap_types(chain: list, swap_set: set[int]) -> list:
    """Rewrite selected hops to Cand_<Op>. `swap_set` is a set of hop-indices."""
    out = []
    for i, (hint, typ, lid, body) in enumerate(chain):
        if i in swap_set:
            out.append((hint, f"Cand_{hint}", lid, body))
        else:
            out.append((hint, typ, lid, body))
    return out


def profile_chain(shape: list[int]) -> dict:
    shape_str = "[" + ",".join(str(x) for x in shape) + "]"
    r = op_profiler.profile_operator(
        "chain", "chain.param", shape_str, threads=1, loop=2000,
        device_dir=DEVDIR, simpleperf_cmd="simpleperf")
    return {"latency_min": r.get("latency_min"), "latency_avg": r.get("latency_avg"),
            "ipc": r.get("ipc"), "error": r.get("error"),
            "trustworthy": r.get("trustworthy")}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="native,ours",
                    help="comma list of: native | ours | swap:<i,j,..>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 1) Install our Cand_<Op> layers for every distinct op in the chain.
    #    Base .cpp comes from runs/<op>/base_kernel/artifacts/; arm .cpp adds NEON.
    distinct = sorted({o for o, *_ in CHAIN_OPS})
    print(f"=== installing candidates: {distinct} ===")
    netoc = NetOracle(ncnn_root=NCNN, workdir=REPO / "opgen/runs/_e2e_chain")
    handles: list = []
    for op in distinct:
        base, arm = base_arm_code(op)
        if not base:
            print(f"  {op}: no base artifacts — abort")
            for h in reversed(handles):
                netoc.restore(h)
            return
        cls = f"Cand_{op}"
        handles.append(netoc.install_layer(base, cls))
        if arm:
            handles.append(netoc.install_layer(arm, cls, subdir="arm", add_cmake=False))
        print(f"  {op}: base+{('arm' if arm else 'base-only')} installed as {cls}")

    results: dict = {"chain": [{"op": o, "type": t, "id": lid, "body": b}
                                for o, t, lid, b in CHAIN_OPS],
                     "input_shape": INPUT_SHAPE, "runs": {}}

    try:
        if not android_rebuild():
            return
        push_benchncnn()

        variants = [v.strip() for v in args.variants.split(",")]
        chain_dir = REPO / "opgen/runs/_e2e_chain"
        chain_dir.mkdir(parents=True, exist_ok=True)

        # write shared BatchNorm weights .bin once
        bin_bytes = build_bin_bytes(CHAIN_OPS)
        bin_path = chain_dir / "chain.bin"
        bin_path.write_bytes(bin_bytes)
        adb("push", str(bin_path), f"{DEVDIR}/chain.bin", timeout=30)

        for v in variants:
            if v == "native":
                chain = CHAIN_OPS
                label = "native"
            elif v == "ours":
                chain = swap_types(CHAIN_OPS, set(range(len(CHAIN_OPS))))
                label = "ours-all"
            elif v.startswith("swap:"):
                idxs = {int(x) for x in v.split(":", 1)[1].split(",") if x}
                chain = swap_types(CHAIN_OPS, idxs)
                label = f"swap-{'.'.join(str(i) for i in sorted(idxs))}"
            else:
                print(f"  unknown variant {v}"); continue

            pfile = chain_dir / f"chain_{label}.param"
            pfile.write_text(build_param(chain))
            adb("push", str(pfile), f"{DEVDIR}/chain.param", timeout=30)
            print(f"\n--- variant={label} ---")
            print(pfile.read_text())
            time.sleep(0.5)
            results["runs"][label] = profile_chain(INPUT_SHAPE)
            print(f"  {label}: {results['runs'][label]}")
    finally:
        for h in reversed(handles):
            netoc.restore(h)
        print("\n=== restored ncnn tree ===")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
