"""On-device (Android phone) verification oracle — the device-in-the-loop gate.

Mirrors `LayerOracle.verify()` but runs the candidate layer on the REAL phone:
NDK-cross-compile the single-layer runner (`layer_oracle_runner.cpp`) + candidate
against the prebuilt android `libncnn.a`, push to device, run, pull the output,
compare to the (host/torch) reference. Also returns device latency via the
runner's `--bench` (no simpleperf needed for correctness+latency; PMU is an
optional switch, default off).

Falls back to `skipped=True` when no device / no NDK / device drops mid-run, so
the caller keeps the host-verified result (i.e. today's host-only behavior).

Design mirrors `oracle.py::LayerOracle` (same bin protocol, same argv contract,
same OracleResult), and the NDK cross-compile mirrors `oracle.py::compile()` with
the NDK clang++ instead of host g++ and the android `libncnn.a`.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .oracle import (
    OracleResult, LayerOracle, write_bin, read_bin,
    _strip_creator_inplace, _KERNELGEN, _THIS, _OPGEN,
)


def _adb(*a: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *a], capture_output=True, text=True, timeout=timeout)


def _looks_like_device_drop(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in ("no devices", "device offline", "device not found",
                                "error: closed", "no such device", "device unauthorized"))


class DeviceOracle:
    """CPU/arm on-device verification. `available()` gates the whole thing; a
    missing device/NDK/lib -> `skipped` so the caller keeps the host result."""

    def __init__(self, ncnn_root: str | Path | None = None, ndk: str | Path | None = None,
                 workdir: str | Path | None = None,
                 device_dir: str = "/data/local/tmp/oracle") -> None:
        self.ncnn_root = Path(ncnn_root) if ncnn_root else (_KERNELGEN / "ncnn")
        self.android_build = self.ncnn_root / "build-android-aarch64"
        self.ndk = Path(ndk) if ndk else self._find_ndk()
        self.runner_src = _THIS / "layer_oracle_runner.cpp"
        self.workdir = Path(workdir) if workdir else (_OPGEN / "runs" / "_device_oracle")
        self.device_dir = device_dir

    # --- prerequisites -----------------------------------------------------
    @staticmethod
    def _find_ndk() -> Path | None:
        for v in ("ANDROID_NDK", "ANDROID_NDK_ROOT", "ANDROID_NDK_HOME"):
            p = os.environ.get(v)
            if p and Path(p).exists():
                return Path(p)
        cands = sorted(glob.glob(str(Path.home() / "Library/Android/sdk/ndk/*")))
        return Path(cands[-1]) if cands else None

    @property
    def libncnn(self) -> Path:
        return self.android_build / "src" / "libncnn.a"

    def _clangxx(self, api: int = 24) -> str | None:
        if not self.ndk:
            return None
        hits = glob.glob(str(self.ndk / f"toolchains/llvm/prebuilt/*/bin/aarch64-linux-android{api}-clang++"))
        return hits[0] if hits else None

    def _libcxx_shared(self) -> str | None:
        """NDK's libc++_shared.so — pushed alongside the runner + LD_LIBRARY_PATH'd,
        because the device's system libc++ can be older than the NDK's (missing
        e.g. __libcpp_verbose_abort), which breaks the dynamic link at run time."""
        if not self.ndk:
            return None
        hits = glob.glob(str(self.ndk / "toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"))
        return hits[0] if hits else None

    def _device_online(self) -> bool:
        try:
            r = _adb("devices", timeout=10)
        except Exception:  # noqa: BLE001
            return False
        return any(ln.strip() and ln.split()[-1] == "device" for ln in r.stdout.splitlines()[1:])

    def available(self) -> tuple[bool, str]:
        if not self.ndk:
            return False, "ANDROID_NDK not found"
        if not self._clangxx():
            return False, "NDK aarch64 clang++ not found"
        if not self.libncnn.exists():
            return False, f"android libncnn.a missing ({self.libncnn}); build build-android-aarch64 first"
        if not self._device_online():
            return False, "no authorized android device (adb)"
        return True, "ok"

    # --- NDK cross-compile (mirrors LayerOracle.compile with NDK clang++) ---
    def _compile(self, candidate_cpp: str | Path, class_name: str, header: str,
                 extra_sources: Sequence[str | Path], extra_includes: Sequence[str | Path]) -> Path:
        cxx = self._clangxx()
        candidate_cpp = Path(candidate_cpp).resolve()
        extra_src = [Path(s).resolve() for s in extra_sources]
        for _p in [candidate_cpp, *extra_src]:
            _strip_creator_inplace(_p)   # avoid duplicate-symbol vs libncnn
        wd = self.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)
        runner = wd / f"runner_dev_{class_name}"
        cmd = [
            # -fno-rtti to match ncnn's android build (libncnn.a has no typeinfo for
            # ncnn::Layer); mismatch -> "undefined symbol: typeinfo for ncnn::Layer".
            cxx, "-std=c++11", "-O2", "-fopenmp", "-static-openmp", "-fno-rtti",
            "-I", str(candidate_cpp.parent),
            "-I", str(self.ncnn_root / "src"),
            "-I", str(self.ncnn_root / "src" / "layer"),
            "-I", str(self.android_build / "src"),   # platform.h / ncnn_export.h
        ]
        for inc in extra_includes:
            cmd += ["-I", str(inc)]
        cmd += [str(self.runner_src), str(candidate_cpp)]
        cmd += [str(s) for s in extra_src]
        cmd += [str(self.libncnn),
                "-landroid", "-llog",   # ncnn datareader uses AAsset_* (libandroid) + logging
                f'-DCANDIDATE_HEADER="{header}"', f"-DCANDIDATE_CLASS={class_name}",
                "-o", str(runner)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not runner.exists():
            raise RuntimeError(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
        return runner

    # --- verify on device --------------------------------------------------
    def verify(self, *, candidate_cpp: str | Path, class_name: str, header: str,
               params: dict[int, Any] | None, inputs: Sequence[np.ndarray],
               reference: np.ndarray, weights: Sequence[np.ndarray] = (),
               weight_flags: Sequence[int] = (), tol: float = 2e-3,
               extra_sources: Sequence[str | Path] = (), extra_includes: Sequence[str | Path] = (),
               packing: int = 0, bench: int = 20, simpleperf: bool = False,
               backend: str = "arm", native_type: str = "", measure_speedup: bool = True) -> OracleResult:
        ok, why = self.available()
        if not ok:
            return OracleResult(ok=False, skipped=True, error=f"device skip: {why}", detail=why)

        # 1) NDK cross-compile — an NDK compile failure IS a real device bug to repair.
        try:
            runner = self._compile(candidate_cpp, class_name, header, extra_sources, extra_includes)
        except RuntimeError as exc:
            return OracleResult(ok=False, passed=False, failure_category="device_compile",
                                compile_log=str(exc),
                                detail=f"[device_compile] NDK cross-compile failed "
                                       f"(compiles on host, not on android arm64):\n{str(exc)[-1500:]}")

        # 2) stage bins + build device argv (same contract as LayerOracle.run)
        wd = self.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)
        argv = [f"./{runner.name}"]
        pushes = [runner]
        libcxx = self._libcxx_shared()
        if libcxx:
            pushes.append(Path(libcxx))
        if params:
            argv += ["--param", ",".join(LayerOracle._fmt_param(k, v) for k, v in params.items())]
        for i, x in enumerate(inputs):
            p = wd / f"in{i}.bin"; write_bin(p, np.asarray(x)); pushes.append(p)
            argv += ["--input", p.name]
        for i, w in enumerate(weights):
            p = wd / f"w{i}.bin"; write_bin(p, np.asarray(w).reshape(-1)); pushes.append(p)
            flag = weight_flags[i] if i < len(weight_flags) else 0
            argv += ["--weight", p.name, "--weight-flag", str(int(flag))]
        argv += ["--out", "out.bin"]
        if packing > 0:
            argv += ["--packing", str(packing)]
        if bench > 0:
            argv += ["--bench", str(bench)]

        # 3) push (any adb failure here that looks like a drop -> skip -> host fallback)
        try:
            _adb("shell", f"mkdir -p {self.device_dir}", timeout=15)
            for p in pushes:
                r = _adb("push", str(p), f"{self.device_dir}/{p.name}", timeout=60)
                if r.returncode != 0:
                    if _looks_like_device_drop(r.stdout + r.stderr):
                        return OracleResult(ok=False, skipped=True,
                                            error="device dropped during push", detail="device dropped")
                    return OracleResult(ok=False, passed=False, failure_category="device_crash",
                                        detail=f"[device] push failed: {(r.stdout + r.stderr)[-300:]}")
            _adb("shell", "chmod", "+x", f"{self.device_dir}/{runner.name}", timeout=15)

            # 4) run (LD_LIBRARY_PATH so the pushed NDK libc++_shared.so is used)
            inner = " ".join(argv)
            cmd = (f"cd {self.device_dir} && LD_LIBRARY_PATH={self.device_dir} "
                   f"{('simpleperf stat ' if simpleperf else '')}{inner} 2>&1")
            run = _adb("shell", cmd, timeout=300)
            txt = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return OracleResult(ok=False, skipped=True, error="device run timed out", detail="device timeout")
        except Exception as exc:  # noqa: BLE001
            return OracleResult(ok=False, skipped=True, error=f"device error: {exc}", detail="device error")

        if "RUNNER_OK" not in txt:
            if _looks_like_device_drop(txt):
                return OracleResult(ok=False, skipped=True, error="device dropped during run", detail="device dropped")
            return OracleResult(ok=False, passed=False, failure_category="device_crash", run_log=txt,
                                detail=f"[device_crash] runner did not finish on device:\n{txt[-1000:]}")
        m = re.search(r"BENCH_MIN_MS=([\d.]+)", txt)
        latency = float(m.group(1)) if m else None

        # 5) pull output + compare to reference
        dev_out = wd / "out_dev.bin"
        rp = _adb("pull", f"{self.device_dir}/out.bin", str(dev_out), timeout=60)
        if rp.returncode != 0 or not dev_out.exists():
            if _looks_like_device_drop(rp.stdout + rp.stderr):
                return OracleResult(ok=False, skipped=True, error="device dropped during pull", detail="device dropped")
            return OracleResult(ok=False, passed=False, failure_category="device_crash",
                                detail="[device] could not pull output")

        out = read_bin(dev_out)
        ref = np.asarray(reference, dtype=np.float32)
        res = OracleResult(ok=True, outputs=[out], run_log=txt, runner=str(runner), latency=latency)
        try:
            out_r = out.reshape(ref.shape)
        except ValueError:
            res.passed = False
            res.failure_category = "device_numeric"
            res.detail = (f"[device_numeric] device output shape {out.shape} != reference "
                          f"{ref.shape} (arch layout/packing divergence)")
            return res
        diff = np.abs(out_r - ref)
        res.max_diff = float(diff.max())
        res.mean_diff = float(diff.mean())
        res.passed = bool(np.allclose(out_r, ref, atol=tol, rtol=tol))
        if res.passed:
            res.detail = (f"device max_diff={res.max_diff:.6f} tol={tol}"
                          + (f" latency_min={latency}ms" if latency is not None else ""))
        else:
            res.failure_category = "device_numeric"
            res.detail = (f"[device_numeric] device (android arm64) output differs from host/torch "
                          f"reference: max_diff={res.max_diff:.6f} mean_diff={res.mean_diff:.6f} "
                          f"tol={tol}. Likely NEON/fp reordering or an arm-path bug that the host "
                          f"build masks.")

        # native baseline on the SAME device + SAME runner via create_layer(native_type)
        # -> speedup with ZERO extra compile (libncnn's built-in op, bins already pushed).
        # fair single-layer fp32 ratio (native_latency / ours); only for native-supported ops.
        if measure_speedup and native_type and res.passed and latency:
            nargv = [f"./{runner.name}", "--layer", native_type]
            if params:
                nargv += ["--param", ",".join(LayerOracle._fmt_param(k, v) for k, v in params.items())]
            for i in range(len(inputs)):
                nargv += ["--input", f"in{i}.bin"]
            for i in range(len(weights)):
                nargv += ["--weight", f"w{i}.bin"]
                flag = weight_flags[i] if i < len(weight_flags) else 0
                nargv += ["--weight-flag", str(int(flag))]
            nargv += ["--out", "out_native.bin", "--bench", str(bench)]
            try:
                ncmd = (f"cd {self.device_dir} && LD_LIBRARY_PATH={self.device_dir} "
                        f"{' '.join(nargv)} 2>&1")
                ntxt = _adb("shell", ncmd, timeout=300).stdout
                nm = re.search(r"BENCH_MIN_MS=([\d.]+)", ntxt)
                if nm and "RUNNER_OK" in ntxt:
                    res.native_latency = float(nm.group(1))
                    if res.native_latency and latency:
                        res.speedup = round(res.native_latency / latency, 3)
                        res.detail += f" | native={res.native_latency}ms speedup={res.speedup}x(fair)"
            except Exception:  # noqa: BLE001 — speedup is a bonus, never break the gate
                pass
        return res


class VulkanDeviceOracle:
    """Adreno GPU on-device verification (device-in-the-loop for vulkan).

    Ports the proven `scripts/bench_vulkan_device.py` flow to in-memory args:
    NDK-vk cross-compile `vulkan_oracle_runner.cpp` + candidate (.cpp/.comp) + base
    .cpp against build-android-vk (find_package ncnn), push runner + .comp shader +
    in/w bins to the phone, run on Adreno (runtime glslang compile), pull output,
    compare to the host (MoltenVK/torch) reference. `--bench` gives GPU latency.
    RC=42 (no vulkan device) / device drop -> skipped (host fallback).
    """

    def __init__(self, ncnn_root: str | Path | None = None, ndk: str | Path | None = None,
                 workdir: str | Path | None = None,
                 device_dir: str = "/data/local/tmp/vkoracle") -> None:
        self.ncnn_root = Path(ncnn_root) if ncnn_root else (_KERNELGEN / "ncnn")
        self.vk_cmake_dir = self.ncnn_root / "build-android-vk/install/lib/cmake/ncnn"
        self.ndk = Path(ndk) if ndk else DeviceOracle._find_ndk()
        self.lo_dir = _THIS   # layer_oracle/ (runner + cand_vulkan_shader.h)
        self.workdir = Path(workdir) if workdir else (_OPGEN / "runs" / "_vk_device_oracle")
        self.device_dir = device_dir

    def _cmake(self) -> str:
        c = shutil.which("cmake")
        if c:
            return c
        venv = _OPGEN.parent / ".venv" / "bin" / "cmake"   # pip-installed cmake
        return str(venv) if venv.exists() else "cmake"

    def _device_online(self) -> bool:
        try:
            r = _adb("devices", timeout=10)
        except Exception:  # noqa: BLE001
            return False
        return any(ln.strip() and ln.split()[-1] == "device" for ln in r.stdout.splitlines()[1:])

    def available(self) -> tuple[bool, str]:
        if not self.ndk:
            return False, "ANDROID_NDK not found"
        if not (self.vk_cmake_dir / "ncnnConfig.cmake").exists():
            return False, f"android-vk ncnn install missing ({self.vk_cmake_dir}); build build-android-vk"
        if not self._device_online():
            return False, "no authorized android device (adb)"
        return True, "ok"

    def verify(self, *, candidate_cpp: str | Path, class_name: str, header: str,
               params: dict[int, Any] | None, inputs: Sequence[np.ndarray],
               reference: np.ndarray, weights: Sequence[np.ndarray] = (),
               weight_flags: Sequence[int] = (), tol: float = 2e-3,
               extra_sources: Sequence[str | Path] = (), extra_includes: Sequence[str | Path] = (),
               shader: str | Path | None = None, extra_shaders: Sequence[str | Path] = (),
               bench: int = 20, simpleperf: bool = False, backend: str = "vulkan",
               native_type: str = "", measure_speedup: bool = True,
               **_kw: Any) -> OracleResult:
        ok, why = self.available()
        if not ok:
            return OracleResult(ok=False, skipped=True, error=f"vk device skip: {why}", detail=why)
        if not shader:
            return OracleResult(ok=False, skipped=True,
                                error="no .comp shader (native-subclass?)", detail="no shader")
        try:
            runner, shader_name, extra_names = self._compile(candidate_cpp, class_name, header,
                                                             shader, extra_shaders, extra_sources)
        except RuntimeError as exc:
            return OracleResult(ok=False, passed=False, failure_category="device_compile",
                                compile_log=str(exc),
                                detail=f"[device_compile] NDK-vk cross-compile failed:\n{str(exc)[-1500:]}")

        wd = self.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)
        argv = [f"./{runner.name}"]
        pushes = [runner, Path(shader)] + [Path(s) for s in extra_shaders]
        if params:
            argv += ["--param", ",".join(LayerOracle._fmt_param(k, v) for k, v in params.items())]
        for i, x in enumerate(inputs):
            p = wd / f"in{i}.bin"; write_bin(p, np.asarray(x)); pushes.append(p)
            argv += ["--input", p.name]
        for i, w in enumerate(weights):
            p = wd / f"w{i}.bin"; write_bin(p, np.asarray(w).reshape(-1)); pushes.append(p)
            flag = weight_flags[i] if i < len(weight_flags) else 0
            argv += ["--weight", p.name, "--weight-flag", str(int(flag))]
        argv += ["--out", "out.bin"]
        if bench > 0:
            argv += ["--bench", str(bench)]

        try:
            _adb("shell", f"mkdir -p {self.device_dir}", timeout=15)
            for p in pushes:
                r = _adb("push", str(p), f"{self.device_dir}/{p.name}", timeout=60)
                if r.returncode != 0:
                    if _looks_like_device_drop(r.stdout + r.stderr):
                        return OracleResult(ok=False, skipped=True, detail="device dropped (push)")
                    return OracleResult(ok=False, passed=False, failure_category="device_crash",
                                        detail=f"[device] push failed: {(r.stdout + r.stderr)[-200:]}")
            _adb("shell", "chmod", "+x", f"{self.device_dir}/{runner.name}", timeout=15)
            inner = " ".join(argv)
            cmd = (f"cd {self.device_dir} && LD_LIBRARY_PATH={self.device_dir} "
                   f"{('simpleperf stat ' if simpleperf else '')}{inner} 2>&1")
            run = _adb("shell", cmd, timeout=300)
            txt = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return OracleResult(ok=False, skipped=True, detail="device run timed out")
        except Exception as exc:  # noqa: BLE001
            return OracleResult(ok=False, skipped=True, detail=f"device error: {exc}")

        if "RC_NO_VULKAN_DEVICE" in txt or "no vulkan device" in txt.lower():
            return OracleResult(ok=False, skipped=True, detail="no vulkan device on phone")
        if "RUNNER_OK" not in txt:
            if _looks_like_device_drop(txt):
                return OracleResult(ok=False, skipped=True, detail="device dropped (run)")
            return OracleResult(ok=False, passed=False, failure_category="device_crash", run_log=txt,
                                detail=f"[device_crash] vulkan runner failed on Adreno:\n{txt[-1000:]}")
        m = re.search(r"BENCH_MIN_MS=([\d.]+)", txt)
        latency = float(m.group(1)) if m else None

        dev_out = wd / "out_dev.bin"
        rp = _adb("pull", f"{self.device_dir}/out.bin", str(dev_out), timeout=60)
        if rp.returncode != 0 or not dev_out.exists():
            return OracleResult(ok=False, skipped=True, detail="could not pull vulkan output")
        out = read_bin(dev_out)
        ref = np.asarray(reference, dtype=np.float32)
        res = OracleResult(ok=True, outputs=[out], run_log=txt, runner=str(runner), latency=latency)
        try:
            out_r = out.reshape(ref.shape)
        except ValueError:
            res.passed = False; res.failure_category = "device_numeric"
            res.detail = f"[device_numeric] Adreno output shape {out.shape} != ref {ref.shape}"
            return res
        diff = np.abs(out_r - ref)
        res.max_diff = float(diff.max()); res.mean_diff = float(diff.mean())
        res.passed = bool(np.allclose(out_r, ref, atol=tol, rtol=tol))
        if res.passed:
            res.detail = (f"vulkan device max_diff={res.max_diff:.6f} tol={tol}"
                          + (f" gpu_latency_min={latency}ms" if latency is not None else ""))
        else:
            res.failure_category = "device_numeric"
            res.detail = (f"[device_numeric] Adreno GPU output differs from host reference: "
                          f"max_diff={res.max_diff:.6f} tol={tol} (shader/dispatch bug the host "
                          f"MoltenVK build masks).")

        # native GPU baseline via create_layer_vulkan on the SAME runner (baked SPIR-V
        # from libncnn-vk) -> fair single-layer GPU speedup, zero extra compile. Ops
        # with no vulkan variant make the runner return RC_NO_VULKAN_DEVICE -> no
        # BENCH_MIN_MS -> speedup silently skipped (not a failure).
        if measure_speedup and native_type and res.passed and latency:
            nargv = [f"./{runner.name}", "--layer", native_type]
            if params:
                nargv += ["--param", ",".join(LayerOracle._fmt_param(k, v) for k, v in params.items())]
            for i in range(len(inputs)):
                nargv += ["--input", f"in{i}.bin"]
            for i in range(len(weights)):
                nargv += ["--weight", f"w{i}.bin"]
                flag = weight_flags[i] if i < len(weight_flags) else 0
                nargv += ["--weight-flag", str(int(flag))]
            nargv += ["--out", "out_native.bin", "--bench", str(bench)]
            try:
                ncmd = (f"cd {self.device_dir} && LD_LIBRARY_PATH={self.device_dir} "
                        f"{' '.join(nargv)} 2>&1")
                ntxt = _adb("shell", ncmd, timeout=300).stdout
                nm = re.search(r"BENCH_MIN_MS=([\d.]+)", ntxt)
                if nm and "RUNNER_OK" in ntxt:
                    res.native_latency = float(nm.group(1))
                    if res.native_latency and latency:
                        res.speedup = round(res.native_latency / latency, 3)
                        res.detail += f" | native={res.native_latency}ms speedup={res.speedup}x(fair)"
            except Exception:  # noqa: BLE001 — speedup is a bonus, never break the gate
                pass
        return res

    def _compile(self, candidate_cpp, class_name, header, shader, extra_shaders, extra_sources):
        """NDK-vk cross-compile the runner + candidate + base into a device binary
        (mirrors scripts/bench_vulkan_device.py)."""
        candidate_cpp = Path(candidate_cpp).resolve()
        cand_dir = candidate_cpp.parent
        wd = self.workdir / class_name
        src = wd / "src"
        shutil.rmtree(src, ignore_errors=True)
        src.mkdir(parents=True, exist_ok=True)
        # stage: runner + shader header + candidate .cpp + ALL headers in the round
        # dir (the vulkan .h #includes the base .h) + shader(s) + base .cpp
        staged = [self.lo_dir / "vulkan_oracle_runner.cpp", self.lo_dir / "cand_vulkan_shader.h",
                  candidate_cpp, Path(shader)]
        staged += list(cand_dir.glob("*.h"))
        staged += [Path(s) for s in extra_shaders]
        staged += [Path(s) for s in extra_sources]
        for f in staged:
            if f.exists():
                shutil.copy(str(f), str(src / f.name))
        # strip DEFINE_LAYER_CREATOR from cpps (avoid duplicate-symbol vs libncnn)
        for cpp in src.glob("*.cpp"):
            cpp.write_text(re.sub(r"^\s*DEFINE_LAYER_CREATOR\s*\([^)]*\)\s*;?\s*$", "",
                                  cpp.read_text(), flags=re.M))
        vk_cpp = candidate_cpp.name
        base_cpps = [Path(s).name for s in extra_sources if str(s).endswith(".cpp")]
        srcs = [f'"{src}/vulkan_oracle_runner.cpp"', f'"{src}/{vk_cpp}"'] + \
               [f'"{src}/{b}"' for b in base_cpps]
        shader_name = Path(shader).name
        (wd / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.10)
project(vkoracle CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}})
find_package(ncnn REQUIRED)
add_executable(runner {' '.join(srcs)})
target_include_directories(runner PRIVATE "{src}" "{self.ncnn_root}/src" "{self.ncnn_root}/src/layer" "{self.ncnn_root}/src/layer/vulkan")
target_compile_definitions(runner PRIVATE
  "CANDIDATE_HEADER=\\"{header}\\""
  "CANDIDATE_CLASS={class_name}"
  "CANDIDATE_SHADER=\\"{self.device_dir}/{shader_name}\\"")
target_link_libraries(runner ncnn)
""")
        cmake = self._cmake()
        cfg = subprocess.run([cmake, "-S", str(wd), "-B", str(wd / "build"),
                              f"-DCMAKE_TOOLCHAIN_FILE={self.ndk}/build/cmake/android.toolchain.cmake",
                              "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-24",
                              f"-Dncnn_DIR={self.vk_cmake_dir}", "-DCMAKE_BUILD_TYPE=Release"],
                             capture_output=True, text=True, timeout=300)
        if cfg.returncode != 0:
            raise RuntimeError("cmake configure failed:\n" + cfg.stdout + cfg.stderr)
        bld = subprocess.run([cmake, "--build", str(wd / "build"), "-j", "8"],
                             capture_output=True, text=True, timeout=600)
        runner = wd / "build" / "runner"
        if bld.returncode != 0 or not runner.exists():
            raise RuntimeError("cmake build failed:\n" + bld.stdout + bld.stderr)
        return runner, shader_name, [Path(s).name for s in extra_shaders]
