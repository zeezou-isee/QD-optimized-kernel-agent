"""VulkanLayerOracle — isolated-instantiation vulkan layer oracle (方案A, vulkan).

Same philosophy as LayerOracle (oracle.py) for base/arm, but for the vulkan
backend: compile a generic GPU runner together with ONE candidate vulkan layer
.cpp + its .comp shader, link against a vulkan-enabled libncnn, then directly
`new <Class>()`, upload inputs to the GPU, run the vulkan forward, download, and
allclose vs PyTorch. It NEVER goes through ncnn::create_layer / Layer_final, so
there is no silent CPU fallback that could mask a broken vulkan kernel.

Two things differ from the base/arm oracle:
  1. The candidate's shader is compiled at RUNTIME (ncnn::compile_spirv_module),
     so we don't depend on the build-time-baked LayerShaderType registry. The
     .comp path is injected as the macro CANDIDATE_SHADER.
  2. Linking a vulkan libncnn pulls in glslang/SPIRV + (on Apple) Metal/Foundation
     frameworks + dl. Rather than hand-roll that g++ link line, we build the runner
     via a tiny generated CMake project that does find_package(ncnn), which carries
     the full transitive link interface for free.

If no Vulkan device is available at run time (e.g. MoltenVK not installed), the
runner exits 42 and the oracle reports `skipped=True` (not a failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import os
import shutil
import subprocess
import sys

import numpy as np

from .oracle import (
    OracleResult,
    write_bin,
    read_bin,
    _find_kernelgen,
    _strip_creator_inplace,
    _run_cmake_bounded,
)
from .failure_taxonomy import classify_failure

_THIS = Path(__file__).resolve().parent           # .../opgen/layer_oracle
_OPGEN = _THIS.parent                              # .../opgen
_KERNELGEN = _find_kernelgen(_THIS)               # .../kernelgen

RC_NO_VULKAN_DEVICE = 42


@dataclass
class VulkanLayerOracle:
    ncnn_root: Path | None = None
    build_lib_vk: Path | None = None
    runner_src: Path | None = None
    cmake: str | None = None
    workdir: Path | None = None

    def __post_init__(self) -> None:
        self.ncnn_root = Path(self.ncnn_root) if self.ncnn_root else (_KERNELGEN / "ncnn")
        self.build_lib_vk = Path(self.build_lib_vk) if self.build_lib_vk else (self.ncnn_root / "build_lib_vk")
        self.runner_src = Path(self.runner_src) if self.runner_src else (_THIS / "vulkan_oracle_runner.cpp")
        self.cmake = self.cmake or shutil.which("cmake") or "cmake"
        self.workdir = Path(self.workdir) if self.workdir else (_OPGEN / "runs" / "_vk_oracle")
        self.workdir.mkdir(parents=True, exist_ok=True)

    # --- prerequisites -----------------------------------------------------
    # We consume ncnn via its INSTALLED cmake package (find_package). The build
    # tree's ncnnConfig.cmake references an export file produced only at install
    # time, so we install to build_lib_vk/install and point ncnn_DIR there.
    @property
    def install_prefix(self) -> Path:
        return self.build_lib_vk / "install"

    @property
    def libncnn(self) -> Path:
        return self.install_prefix / "lib" / "libncnn.a"

    @property
    def ncnn_dir(self) -> Path:
        return self.install_prefix / "lib" / "cmake" / "ncnn"

    def _ensure_libncnn(self) -> None:
        if self.libncnn.exists() and (self.ncnn_dir / "ncnnConfig.cmake").exists():
            return
        raise FileNotFoundError(
            f"vulkan ncnn package not found at {self.ncnn_dir}. Build + install it once with:\n"
            f"  git -C {self.ncnn_root} submodule update --init glslang\n"
            f"  cmake -S {self.ncnn_root} -B {self.build_lib_vk} "
            f"-DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_BUILD_TOOLS=OFF "
            f"-DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF "
            f"-DNCNN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release\n"
            f"  cmake --build {self.build_lib_vk} -j\n"
            f"  cmake --install {self.build_lib_vk} --prefix {self.install_prefix}"
        )

    # --- compile (cached by input mtime) -----------------------------------
    def compile(self, candidate_cpp: str | Path, class_name: str, header: str,
                shader: str | Path | None,
                extra_sources: Sequence[str | Path] = (),
                extra_includes: Sequence[str | Path] = ()) -> tuple[Path, str]:
        """Build runner + candidate.cpp [+ extra_sources] against vulkan libncnn.

        `shader` is the candidate's .comp file (injected as CANDIDATE_SHADER and
        compiled at runtime) for the FROM-SCRATCH path. Pass None for the
        NATIVE-SUBCLASS path (`Cand_X_vulkan : public ncnn::X_vulkan`), which
        inherits ncnn's built-in create_pipeline + baked SPIR-V and needs no
        candidate shader. The runner exe path is returned.
        """
        self._ensure_libncnn()
        candidate_cpp = Path(candidate_cpp).resolve()
        shader = Path(shader).resolve() if shader is not None else None
        extra_src = [Path(s).resolve() for s in extra_sources]
        for _p in [candidate_cpp, *extra_src]:
            _strip_creator_inplace(_p)

        proj = self.workdir / f"proj_{class_name}"
        proj.mkdir(parents=True, exist_ok=True)
        build = proj / "build"
        runner = build / "runner"

        # rebuild if exe missing or any input newer than exe
        inputs = [candidate_cpp, self.runner_src, _THIS / "cand_vulkan_shader.h"]
        if shader is not None:
            inputs.append(shader)
        inputs += [s for s in extra_src if s.exists()]
        newest = max(p.stat().st_mtime for p in inputs if p.exists())
        if runner.exists() and runner.stat().st_mtime >= newest:
            return runner, "(cached)"

        cml = self._cmakelists(candidate_cpp, class_name, header, shader, extra_src, extra_includes)
        (proj / "CMakeLists.txt").write_text(cml, encoding="utf-8")

        cfg = [self.cmake, "-S", str(proj), "-B", str(build),
               f"-Dncnn_DIR={self.ncnn_dir}", "-DCMAKE_BUILD_TYPE=Release"]
        p1 = subprocess.run(cfg, capture_output=True, text=True)
        log = " ".join(cfg) + "\n" + p1.stdout + p1.stderr
        if p1.returncode != 0:
            raise RuntimeError(f"vulkan runner cmake configure failed:\n{log}")

        bld = [self.cmake, "--build", str(build), "-j", "8"]
        # bounded + own session so a parent SIGTERM kills cmake/make/g++ cleanly.
        rc2, out2 = _run_cmake_bounded(bld, timeout=600)
        log += "\n" + " ".join(bld) + "\n" + out2
        if rc2 != 0 or not runner.exists():
            raise RuntimeError(f"vulkan runner build failed:\n{log}")
        return runner, log

    def _cmakelists(self, candidate_cpp: Path, class_name: str, header: str,
                    shader: Path | None, extra_src: list[Path], extra_includes: Sequence[str | Path]) -> str:
        srcs = " ".join(f'"{s}"' for s in [self.runner_src, candidate_cpp, *extra_src])
        # ncnn source-tree includes so a NATIVE-SUBCLASS candidate can
        # `#include "vulkan/<op>_vulkan.h"` (internal layer headers are NOT
        # exported by the install, only present in the source tree). Harmless for
        # the from-scratch path.
        src = self.ncnn_root / "src"
        incs = " ".join(f'"{i}"' for i in [
            candidate_cpp.parent,            # candidate's own header
            _THIS,                           # cand_vulkan_shader.h
            src, src / "layer", src / "layer" / "vulkan",
            *[Path(x) for x in extra_includes],
        ])
        shader_def = f'\n    "CANDIDATE_SHADER=\\"{shader}\\""' if shader is not None else ""
        return f"""cmake_minimum_required(VERSION 3.10)
project(vk_oracle CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}})
find_package(ncnn REQUIRED)
add_executable(runner {srcs})
target_include_directories(runner PRIVATE {incs})
target_compile_definitions(runner PRIVATE
    "CANDIDATE_HEADER=\\"{header}\\""
    "CANDIDATE_CLASS={class_name}"{shader_def})
target_link_libraries(runner ncnn)
"""

    # --- run ---------------------------------------------------------------
    def run(self, *, candidate_cpp: str | Path, class_name: str, header: str,
            shader: str | Path | None = None,
            params: dict[int, Any] | None = None,
            inputs: Sequence[np.ndarray],
            weights: Sequence[np.ndarray] = (),
            weight_flags: Sequence[int] = (),
            extra_sources: Sequence[str | Path] = (),
            extra_includes: Sequence[str | Path] = (),
            packing: int = 0) -> OracleResult:
        try:
            runner, clog = self.compile(candidate_cpp, class_name, header, shader,
                                        extra_sources=extra_sources, extra_includes=extra_includes)
        except Exception as exc:  # noqa: BLE001
            return OracleResult(ok=False, error=str(exc), compile_log=str(exc))

        wd = self.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)

        argv = [str(runner)]
        if params:
            argv += ["--param", ",".join(self._fmt_param(k, v) for k, v in params.items())]
        for i, x in enumerate(inputs):
            p = wd / f"in{i}.bin"
            write_bin(p, np.asarray(x))
            argv += ["--input", str(p)]
        for i, w in enumerate(weights):
            p = wd / f"w{i}.bin"
            write_bin(p, np.asarray(w).reshape(-1))
            argv += ["--weight", str(p)]
            flag = weight_flags[i] if i < len(weight_flags) else 0
            argv += ["--weight-flag", str(int(flag))]
        out_path = wd / "out.bin"
        argv += ["--out", str(out_path)]
        if packing > 0:
            argv += ["--packing", str(packing)]

        proc = subprocess.run(argv, capture_output=True, text=True, env=self._runner_env())
        run_log = " ".join(argv) + "\n" + proc.stdout + proc.stderr

        if proc.returncode == RC_NO_VULKAN_DEVICE:
            return OracleResult(ok=False, skipped=True, return_code=proc.returncode,
                                compile_log=clog, run_log=run_log, runner=str(runner),
                                error="no vulkan device (skipped)")
        if proc.returncode != 0 or not out_path.exists():
            return OracleResult(ok=False, return_code=proc.returncode, compile_log=clog,
                                run_log=run_log, runner=str(runner),
                                error=f"vulkan runner failed (rc={proc.returncode})")
        out = read_bin(out_path)
        return OracleResult(ok=True, outputs=[out], return_code=0, compile_log=clog,
                            run_log=run_log, runner=str(runner))

    # --- verify vs reference (oracle) --------------------------------------
    def verify(self, *, candidate_cpp: str | Path, class_name: str, header: str,
               shader: str | Path | None = None,
               params: dict[int, Any] | None,
               inputs: Sequence[np.ndarray],
               reference: np.ndarray,
               weights: Sequence[np.ndarray] = (),
               weight_flags: Sequence[int] = (),
               tol: float = 1e-3,
               extra_sources: Sequence[str | Path] = (),
               extra_includes: Sequence[str | Path] = (),
               packing: int = 0,
               backend: str = "vulkan") -> OracleResult:
        res = self.run(candidate_cpp=candidate_cpp, class_name=class_name, header=header,
                       shader=shader, params=params, inputs=inputs, weights=weights,
                       weight_flags=weight_flags,
                       extra_sources=extra_sources, extra_includes=extra_includes, packing=packing)
        if res.skipped:
            res.passed = None
            res.detail = "vulkan device unavailable (skipped)"
            return res
        if not res.ok:
            res.passed = False
            res.detail = res.error or "runner did not produce output"
            return res
        out = res.outputs[0]
        ref = np.asarray(reference, dtype=np.float32)
        _inp = inputs[0] if len(inputs) else None
        try:
            out_r = out.reshape(ref.shape)
        except ValueError:
            cat, det = classify_failure(out, ref, tol, input=_inp, backend=backend)
            res.passed = False
            res.failure_category = cat
            res.detail = f"[{cat}] {det}"
            return res
        diff = np.abs(out_r - ref)
        res.max_diff = float(diff.max())
        res.mean_diff = float(diff.mean())
        res.passed = bool(np.allclose(out_r, ref, atol=tol, rtol=tol))
        if res.passed:
            res.detail = f"max_diff={res.max_diff:.6f} mean_diff={res.mean_diff:.6f} tol={tol}"
        else:
            cat, det = classify_failure(out, ref, tol, input=_inp, backend=backend)
            res.failure_category = cat
            res.detail = f"[{cat}] {det}"
        return res

    @staticmethod
    def _runner_env() -> dict[str, str]:
        """ncnn's simplevk loader reads NCNN_VULKAN_DRIVER to dlopen an ICD directly.
        On macOS there is no system libvulkan, so if the user hasn't pointed it at a
        driver, auto-detect a MoltenVK install (brew) so the runner finds the GPU.
        """
        env = dict(os.environ)
        if sys.platform == "darwin" and not env.get("NCNN_VULKAN_DRIVER"):
            for p in ("/opt/homebrew/lib/libMoltenVK.dylib",
                      "/usr/local/lib/libMoltenVK.dylib"):
                if Path(p).exists():
                    env["NCNN_VULKAN_DRIVER"] = p
                    break
        return env

    @staticmethod
    def _fmt_param(key: int, value: Any) -> str:
        if isinstance(value, float):
            return f"{key}={value:.8g}" if value == value else f"{key}=0"
        return f"{key}={int(value)}"
