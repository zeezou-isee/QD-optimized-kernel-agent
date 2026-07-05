"""P1 — from-scratch vulkan kernels on the REAL phone GPU (Adreno), batched.

For each op with a from-scratch vulkan kernel: cross-compile the vulkan oracle
runner + candidate against android-vk libncnn (glslang baked in → runtime shader
compile on device), push runner + .comp + the host oracle's input/weight bins,
run on the Adreno GPU with --bench (device latency) + compare output to the host
MoltenVK reference (already verified == torch). Records correctness + GPU latency.

Reuses opgen/runs/_vk_oracle/Cand_<Op>_vulkan/{in*.bin,w*.bin,out.bin,argv.txt}.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/xingze/Documents/project/kernelgen/QD-optimized-kernel-agent")
NCNN = Path("/Users/xingze/Documents/project/kernelgen/ncnn")
NCNN_DIR = NCNN / "build-android-vk/install/lib/cmake/ncnn"
NDK = Path.home() / "Library/Android/sdk/ndk/29.0.13846066"
LO = REPO / "opgen/layer_oracle"
DEVDIR = "/data/local/tmp/vkrun"
OUT = REPO / "batch/results/miniset_vulkan_device.json"
MINISET = ["Abs", "Add", "And", "AveragePool", "BatchNormalization",
           "Conv", "Gemm", "Greater", "Mul", "ReduceSum"]  # ReduceMax: no host artifacts


def sh(cmd, **k):
    return subprocess.run(cmd, capture_output=True, text=True, **k)


def read_bin(p: Path):
    import numpy as np, struct
    with open(p, "rb") as f:
        ndim = struct.unpack("i", f.read(4))[0]
        dims = struct.unpack(f"{ndim}i", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype="float32")
    return data.reshape(dims) if ndim else data


def parse_argv(txt: str):
    """Extract (param, inputs[], weights[], weight_flags[]) from host argv.txt."""
    toks = txt.split()
    param, inputs, weights, wflags = "", [], [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "--param" and i + 1 < len(toks): param = toks[i + 1]; i += 2
        elif t == "--input" and i + 1 < len(toks): inputs.append(Path(toks[i + 1]).name); i += 2
        elif t == "--weight" and i + 1 < len(toks): weights.append(Path(toks[i + 1]).name); i += 2
        elif t == "--weight-flag" and i + 1 < len(toks): wflags.append(toks[i + 1]); i += 2
        else: i += 1
    return param, inputs, weights, wflags


def candidate_dir(op: str) -> Path | None:
    for d in sorted((REPO / "opgen/runs" / op / "backends/vulkan/kernel").glob("round_*"), reverse=True):
        if list(d.glob("*.comp")):
            return d
    return None


def bench_op(op: str, iters: int) -> dict:
    host = REPO / "opgen/runs/_vk_oracle" / f"Cand_{op}_vulkan"
    cand = candidate_dir(op)
    prof_p = REPO / "opgen/runs" / op / "backends/vulkan/kernel/kernel_profile.json"
    if not host.exists() or not cand or not prof_p.exists():
        return {"op": op, "error": f"missing host={host.exists()} cand={cand} prof={prof_p.exists()}"}
    prof = json.loads(prof_p.read_text())
    if prof.get("native_vulkan"):
        return {"op": op, "error": "native-subclass (no from-scratch shader) — skip"}
    cls = prof["class_name"]; header = prof["header"]; shader = prof["shader"]
    param, inputs, weights, wflags = parse_argv((host / "argv.txt").read_text())
    art = REPO / "opgen/runs" / op / "base_kernel/artifacts"

    work = Path("/tmp") / f"vkrun_{op}"
    sh(["rm", "-rf", str(work)]); (work / "src").mkdir(parents=True)
    # stage candidate + runner + base, strip creators
    for f in [LO / "vulkan_oracle_runner.cpp", LO / "cand_vulkan_shader.h"]:
        sh(["cp", str(f), str(work / "src")])
    for f in cand.glob("*"):
        if f.suffix in (".h", ".cpp", ".comp"):
            sh(["cp", str(f), str(work / "src")])
    for f in art.glob("*"):
        if f.suffix in (".h", ".cpp"):
            sh(["cp", str(f), str(work / "src")])
    for cpp in work.glob("src/*.cpp"):
        txt = re.sub(r"^\s*DEFINE_LAYER_CREATOR\s*\([^)]*\)\s*;?\s*$", "", cpp.read_text(), flags=re.M)
        cpp.write_text(txt)
    cand_cpp = f"{cls[len('Cand_'):].lower()}"  # e.g. Cand_Abs_vulkan -> abs... not robust
    # source list: runner + candidate vulkan cpp + base cpp
    vk_cpp = next((p.name for p in (work / "src").glob("*_vulkan.cpp")), None)
    base_cpp = next((p.name for p in (work / "src").glob("*.cpp")
                     if not p.name.endswith("_vulkan.cpp") and p.name != "vulkan_oracle_runner.cpp"), None)
    srcs = [f'"{work}/src/vulkan_oracle_runner.cpp"', f'"{work}/src/{vk_cpp}"']
    if base_cpp:
        srcs.append(f'"{work}/src/{base_cpp}"')
    (work / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.10)
project(vkrun CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}})
find_package(ncnn REQUIRED)
add_executable(runner {' '.join(srcs)})
target_include_directories(runner PRIVATE "{work}/src" "{NCNN}/src" "{NCNN}/src/layer" "{NCNN}/src/layer/vulkan")
target_compile_definitions(runner PRIVATE
  "CANDIDATE_HEADER=\\"{header}\\""
  "CANDIDATE_CLASS={cls}"
  "CANDIDATE_SHADER=\\"{DEVDIR}/{shader}\\"")
target_link_libraries(runner ncnn)
""")
    # configure + build
    cfg = sh(["cmake", "-S", str(work), "-B", str(work / "build"),
              f"-DCMAKE_TOOLCHAIN_FILE={NDK}/build/cmake/android.toolchain.cmake",
              "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-24",
              f"-Dncnn_DIR={NCNN_DIR}", "-DCMAKE_BUILD_TYPE=Release"], timeout=300)
    if cfg.returncode != 0:
        return {"op": op, "error": "configure failed", "log": cfg.stderr[-400:]}
    bld = sh(["cmake", "--build", str(work / "build"), "-j", "8"], timeout=600)
    if bld.returncode != 0 or not (work / "build/runner").exists():
        return {"op": op, "error": "build failed", "log": bld.stderr[-600:]}

    # push runner + shader + bins
    sh(["adb", "shell", f"mkdir -p {DEVDIR}"])
    sh(["adb", "push", str(work / "build/runner"), f"{DEVDIR}/runner"])
    sh(["adb", "push", str(work / "src" / shader), f"{DEVDIR}/{shader}"])
    for b in inputs + weights:
        sh(["adb", "push", str(host / b), f"{DEVDIR}/{b}"])
    sh(["adb", "shell", "chmod", "+x", f"{DEVDIR}/runner"])

    # device argv
    argv = f"cd {DEVDIR} && ./runner"
    if param: argv += f" --param {param}"
    for b in inputs: argv += f" --input {b}"
    for j, b in enumerate(weights):
        argv += f" --weight {b}"
        if j < len(wflags): argv += f" --weight-flag {wflags[j]}"
    argv += f" --out out.bin --bench {iters} 2>&1"
    run = sh(["adb", "shell", argv], timeout=180)
    txt = run.stdout + run.stderr
    m = re.search(r"BENCH_MIN_MS=([\d.]+)", txt)
    lat = float(m.group(1)) if m else None
    ok = "RUNNER_OK" in txt

    # correctness vs host reference
    max_diff = None
    if ok:
        sh(["adb", "pull", f"{DEVDIR}/out.bin", str(work / "out_dev.bin")])
        try:
            import numpy as np
            dev = read_bin(work / "out_dev.bin"); ref = read_bin(host / "out.bin")
            max_diff = float(np.abs(dev.reshape(ref.shape) - ref).max())
        except Exception as e:  # noqa: BLE001
            max_diff = f"compare failed: {e}"
    return {"op": op, "ran": ok, "gpu_latency_min_ms": lat, "max_diff_vs_host": max_diff,
            "tail": txt[-300:] if not ok else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=None)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",")] if args.ops else MINISET
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    for op in ops:
        print(f"\n=== vulkan device: {op} ===")
        try:
            r = bench_op(op, args.iters)
        except Exception as exc:  # noqa: BLE001
            r = {"op": op, "error": f"crashed: {exc}"}
        print(f"  ran={r.get('ran')} gpu_min={r.get('gpu_latency_min_ms')} max_diff={r.get('max_diff_vs_host')} err={r.get('error','')}")
        allres[op] = r
        OUT.write_text(json.dumps(allres, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    for op in ops:
        r = allres.get(op, {})
        print(f"{op:<20} gpu={r.get('gpu_latency_min_ms')} ms  correct(max_diff)={r.get('max_diff_vs_host')}  {r.get('error','')}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
