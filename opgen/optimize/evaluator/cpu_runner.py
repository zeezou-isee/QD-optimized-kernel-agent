"""Single-shot CPU runner — compile a materialized kernel and run it ONCE.

Built on LayerOracle (opgen/layer_oracle/oracle.py). One run = one compiled
binary + one forward pass writing output.bin. We do NOT recompile per
repetition: the measure harness invokes the cached binary directly N times
(see measure_harness.py for the timing loop).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from layer_oracle import LayerOracle, read_bin, write_bin


@dataclass
class RunArtifacts:
    """What CpuRunner.compile_only produces — handed to the measure harness."""
    runner_path: Path
    inputs_bins: list[Path]
    weights_bins: list[Path]
    out_bin: Path
    params_argv: list[str]            # ["--param", "0=4,1=3"] or []
    packing: int = 0                  # arm NC4HW4 elempack (0 = off)
    fp16_storage: bool = False        # runner: opt.use_fp16_packed/storage
    fp16_arith: bool = False          # runner: opt.use_fp16_arithmetic (needs HAS_ASIMDHP)


class CpuRunner:
    """Thin wrapper around LayerOracle for the optimize loop.

    Two responsibilities:
      1. compile_only(): compile a kernel candidate + write input/weight bins;
         returns artifacts the harness can invoke N times.
      2. run_once(): execute the runner binary once and read output (no
         compile, no I/O re-prep) — used by the correctness oracle.
    """

    def __init__(self, oracle: LayerOracle) -> None:
        self.oracle = oracle

    # ----- compile + I/O prep (one-shot) -------------------------------------
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
        packing: int = 0,
        fp16_storage: bool = False,   # arm fp16-storage tier (halves bytes moved)
        fp16_arith: bool = False,     # arm fp16-arith tier (needs ARMv8.2 FP16 / HAS_ASIMDHP)
        shader: Path | None = None,   # ignored by CpuRunner (vulkan-only); kept for a uniform call
    ) -> tuple[RunArtifacts, str]:
        """Compile the candidate kernel + lay out the I/O .bin files.

        For arm: pass the verified base .cpp via `extra_sources`, `src/layer/arm`
        via `extra_includes`, and `packing=4` (recorded so each run uses NC4HW4).
        Returns (artifacts, compile_log). Raises on compile failure.
        """
        runner, clog = self.oracle.compile(candidate_cpp, class_name, header,
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
                            params_argv=params_argv, packing=packing,
                            fp16_storage=fp16_storage, fp16_arith=fp16_arith), clog

    # ----- single forward pass (no compile) ---------------------------------
    def run_once(self, art: RunArtifacts, timeout_s: float = 30.0) -> tuple[bool, float, str]:
        """Run the cached binary once; return (ok, elapsed_ms, stderr_tail).

        The elapsed time is wall-clock around the subprocess (process startup
        included). For very fast kernels this measurement is dominated by
        subprocess overhead — that's accepted at M1; M2 can switch to an
        in-runner timing loop if needed.
        """
        argv = [str(art.runner_path)]
        argv += art.params_argv
        for p in art.inputs_bins:
            argv += ["--input", str(p)]
        for p in art.weights_bins:
            argv += ["--weight", str(p)]
        argv += ["--out", str(art.out_bin)]
        if art.packing > 0:
            argv += ["--packing", str(art.packing)]
        if art.fp16_arith:
            argv += ["--fp16-arith"]
        elif art.fp16_storage:
            argv += ["--fp16-storage"]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, float("inf"), f"timeout > {timeout_s}s"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ok = proc.returncode == 0 and art.out_bin.exists()
        err = "" if ok else (proc.stderr or "")[-300:]
        return ok, elapsed_ms, err

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _fmt_param(key: int, value) -> str:
        if isinstance(value, float):
            return f"{key}={value:.8g}"
        return f"{key}={int(value)}"

    @staticmethod
    def read_output(art: RunArtifacts) -> np.ndarray:
        return read_bin(art.out_bin)
