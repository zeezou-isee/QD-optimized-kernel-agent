"""Single-shot VULKAN runner for the optimize loop — mirror of CpuRunner.

Same contract as CpuRunner (compile_only -> RunArtifacts; run_once; read_output)
so the MeasureHarness / Evaluator drive it identically, but it compiles & runs a
vulkan candidate via VulkanLayerOracle (isolated instantiation on the GPU, the
`.comp` shader compiled at runtime). One run = one cached binary + one GPU forward.

No Vulkan device (e.g. no MoltenVK): the runner exits 42 -> run_once returns a
skip-flagged failure, so the optimizer degrades gracefully (baseline measurement
fails -> optimization is skipped, same as a missing device elsewhere).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from layer_oracle import VulkanLayerOracle, read_bin, write_bin

from .cpu_runner import RunArtifacts

RC_NO_VULKAN_DEVICE = 42


class VkRunner:
    """Thin wrapper around VulkanLayerOracle, drop-in for CpuRunner."""

    def __init__(self, oracle: VulkanLayerOracle) -> None:
        self.oracle = oracle

    def compile_only(
        self,
        *,
        candidate_cpp: Path,
        class_name: str,
        header: str,
        inputs: Sequence[np.ndarray],
        weights: Sequence[np.ndarray] = (),
        params: dict[int, object] | None = None,
        extra_sources: Sequence[Path] = (),
        extra_includes: Sequence[Path] = (),
        packing: int = 0,             # ignored (v1 vulkan runs elempack=1)
        shader: Path | None = None,
    ) -> tuple[RunArtifacts, str]:
        if shader is None:
            raise ValueError("vulkan candidate requires a .comp shader (shader=None)")
        runner, clog = self.oracle.compile(candidate_cpp, class_name, header, shader,
                                           extra_sources=extra_sources, extra_includes=extra_includes)
        wd = self.oracle.workdir / class_name
        wd.mkdir(parents=True, exist_ok=True)

        argv_in: list[Path] = []
        for i, x in enumerate(inputs):
            p = wd / f"in{i}.bin"
            write_bin(p, np.asarray(x))
            argv_in.append(p)
        argv_w: list[Path] = []
        for i, w in enumerate(weights):
            p = wd / f"w{i}.bin"
            write_bin(p, np.asarray(w).reshape(-1))
            argv_w.append(p)

        out = wd / "out.bin"
        params_argv: list[str] = []
        if params:
            params_argv = ["--param", ",".join(self._fmt_param(k, v) for k, v in params.items())]
        return RunArtifacts(runner_path=Path(runner), inputs_bins=argv_in,
                            weights_bins=argv_w, out_bin=out,
                            params_argv=params_argv, packing=0), clog

    def run_once(self, art: RunArtifacts, timeout_s: float = 60.0) -> tuple[bool, float, str]:
        argv = [str(art.runner_path)] + art.params_argv
        for p in art.inputs_bins:
            argv += ["--input", str(p)]
        for p in art.weights_bins:
            argv += ["--weight", str(p)]
        argv += ["--out", str(art.out_bin)]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                                  env=self.oracle._runner_env())   # auto-detects MoltenVK on macOS
        except subprocess.TimeoutExpired:
            return False, float("inf"), f"timeout > {timeout_s}s"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode == RC_NO_VULKAN_DEVICE:
            return False, float("inf"), "no vulkan device (skipped)"
        ok = proc.returncode == 0 and art.out_bin.exists()
        err = "" if ok else (proc.stderr or "")[-300:]
        return ok, elapsed_ms, err

    @staticmethod
    def _fmt_param(key: int, value) -> str:
        if isinstance(value, float):
            return f"{key}={value:.8g}"
        return f"{key}={int(value)}"

    @staticmethod
    def read_output(art: RunArtifacts) -> np.ndarray:
        return read_bin(art.out_bin)
