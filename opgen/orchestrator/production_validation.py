"""Production-grade validation: compile + correctness + (optional) android benchmark.

Mirrors what MoKA's NCNN pipeline does after register, but plugged after our
OperatorAgent's end-to-end numeric so we keep the existing flow intact:

  production_compile     build_lib (default; quick - uses existing libncnn.a) OR
                         build_full (MoKA-style full ncnn build with examples+tests)
  production_correctness reuse NetOracle: run converted .ncnn.{param,bin} vs PyTorch
  benchmark              MoKA-style android arm64 cross-build + adb push benchncnn +
                         parse min/max/avg. Skipped (NOT failed) if no device/NDK.

All three are independent: failures in must-have steps fail the operator;
benchmark failure only annotates the summary.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_oracle import (
    NetOracle,
    parse_ncnn_io,
    pnnx_driven_ncnn_inputs,
    retarget_param_output_layer,
    retarget_param_output_file,
)
from layer_oracle.oracle import _run_cmake_bounded


# ---------------------------------------------------------------------------
def detect_android_device() -> tuple[bool, str]:
    """Return (have_device, reason). Skipped reasons: no adb / no device."""
    try:
        proc = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return False, "adb not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "adb devices timed out"
    if proc.returncode != 0:
        return False, f"adb devices failed (rc={proc.returncode})"
    devices = [ln.split("\t", 1)[0] for ln in proc.stdout.splitlines()[1:]
               if "\tdevice" in ln]
    if not devices:
        return False, "no authorized android device attached"
    return True, ",".join(devices)


def parse_benchmark_output(log_text: str) -> dict[str, Any]:
    """Adapted from MoKA's parse_benchmark_output (ncnnpipeline/ncnn_pipeline.py)."""
    perf_re = re.compile(
        r"\s+(?P<m>\S+)\s+min\s*=\s*(?P<mn>[\d.]+)\s+max\s*=\s*(?P<mx>[\d.]+)\s+avg\s*=\s*(?P<av>[\d.]+)"
    )
    m = perf_re.search(log_text)
    loop = re.search(r"loop_count\s*=\s*(\d+)", log_text)
    threads = re.search(r"num_threads\s*=\s*(\d+)", log_text)
    if m:
        return {
            "runloop": int(loop.group(1)) if loop else 100,
            "num_threads": int(threads.group(1)) if threads else 4,
            "min": float(m.group("mn")), "max": float(m.group("mx")), "avg": float(m.group("av")),
            "error": "",
        }
    return {"runloop": 100, "num_threads": 4, "min": None, "max": None, "avg": None,
            "error": log_text[-400:]}


# ---------------------------------------------------------------------------
@dataclass
class ProductionValidator:
    ncnn_root: Path
    compile_mode: str = "build_lib"        # "build_lib" | "build_full"
    do_benchmark: bool = False
    workdir: Path = field(default_factory=Path)  # runs/<task>/operator
    android_ndk: str | None = None
    build_jobs: int = 8

    def __post_init__(self) -> None:
        self.ncnn_root = Path(self.ncnn_root)
        if self.android_ndk is None:
            self.android_ndk = os.environ.get("ANDROID_NDK")
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    # --- 1. compile -------------------------------------------------------
    @property
    def libncnn_a(self) -> Path:
        return self.ncnn_root / "build_lib" / "src" / "libncnn.a"

    @property
    def build_full_dir(self) -> Path:
        return self.ncnn_root / "build_full"

    def production_compile(self) -> dict:
        out: dict = {"mode": self.compile_mode, "ok": False, "log_tail": ""}
        if self.compile_mode == "build_lib":
            if not self.libncnn_a.exists():
                out["log_tail"] = f"libncnn.a missing at {self.libncnn_a}; bridge step likely failed"
                return out
            out["ok"] = True
            out["log_tail"] = f"libncnn.a present ({self.libncnn_a.stat().st_size} bytes)"
            return out

        if self.compile_mode == "build_full":
            log_path = self.workdir / "production_compile.log"
            self.build_full_dir.mkdir(parents=True, exist_ok=True)
            cfg_cmd = ["cmake", "-S", str(self.ncnn_root), "-B", str(self.build_full_dir),
                       "-DCMAKE_BUILD_TYPE=Release",
                       "-DNCNN_BUILD_EXAMPLES=ON", "-DNCNN_BUILD_TESTS=ON",
                       "-DNCNN_BUILD_BENCHMARK=ON",
                       "-DNCNN_VULKAN=OFF", "-DNCNN_PYTHON=OFF"]
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(cfg_cmd) + "\n")
                cfg = subprocess.run(cfg_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if cfg.returncode != 0:
                out["log_tail"] = "[configure failed]\n" + log_path.read_text(errors="replace")[-600:]
                return out
            build_cmd = ["cmake", "--build", str(self.build_full_dir), "-j", str(self.build_jobs)]
            # bounded + own session so a parent SIGTERM kills make/g++ cleanly
            rc, output = _run_cmake_bounded(build_cmd, timeout=900)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("$ " + " ".join(build_cmd) + "\n" + output)
            out["log_tail"] = log_path.read_text(errors="replace")[-800:]
            out["ok"] = rc == 0
            return out

        out["log_tail"] = f"unknown compile_mode={self.compile_mode}"
        return out

    # --- 2. correctness (NetOracle reuse) --------------------------------
    def production_correctness(self, graph_sum: dict, model_py: str | Path,
                               tol: float = 2e-3, retarget_to: str | None = None,
                               expected_src_type: str | None = None) -> dict:
        art = (graph_sum.get("final_result") or {}).get("artifacts") or {}
        param, binf = art.get(".ncnn.param"), art.get(".ncnn.bin")
        if not param or not binf or not Path(param).exists():
            return {"passed": False, "detail": "no converted .ncnn.param/.bin in graph summary"}

        # re-point the output layer to OUR impl so the net runs ours, not built-in
        # (needed for ops ncnn already supports; idempotent for new ops). The
        # expected_src_type guard skips the retarget for decomposed ops (output
        # layer is a different native type) so the baseline native graph runs.
        if retarget_to:
            rp = self.workdir / "prod_correctness_retargeted.param"
            retarget_param_output_file(param, rp, retarget_to,
                                       expected_src_type=expected_src_type)
            param = str(rp)

        import torch
        spec = importlib.util.spec_from_file_location("ds_model", str(model_py))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)  # identical weights to the exported bin (make_pt)
        model = (mod.Model(*init) if init else mod.Model()).eval()
        inputs = mod.get_inputs()
        with torch.no_grad():
            ref = model(*inputs)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        ref_np = ref.detach().numpy()
        in_names, out_name = parse_ncnn_io(Path(param).read_text(encoding="utf-8"))
        if len(in_names) != len(inputs):
            in_names = [f"in{i}" for i in range(len(inputs))]
        # pnnx-driven per-blob squeeze policy (falls back to drop-axis-0 when
        # _ncnn.py is missing or a blob name has no recorded policy).
        ncnn_py = art.get("_ncnn.py")
        ncnn_inputs = pnnx_driven_ncnn_inputs(inputs[:len(in_names)], in_names, ncnn_py)
        feed = {n: x for n, x in zip(in_names, ncnn_inputs)}
        # Reference shape must mirror the squeeze pnnx applied to in0: drop axis 0
        # only when in0 was actually squeezed (nn.Module batch dim). For ops like
        # Gemm where axis 0 is M (not batch) pnnx keeps the full rank, so keep the
        # full reference. (Same logic as operator_agent._net_numeric_impl.)
        in0_squeezed = (inputs[0].ndim >= 2
                        and ncnn_inputs[0].ndim == inputs[0].ndim - 1)
        reference = ref_np[0] if (in0_squeezed and ref_np.ndim >= 2) else ref_np

        netoc = NetOracle(ncnn_root=self.ncnn_root, workdir=self.workdir / "_prod_net")
        out, log = netoc.run_net(param, binf, feed, out_name)
        (self.workdir / "production_net.log").write_text(log, encoding="utf-8")
        if out is None:
            return {"passed": False, "detail": "net runner failed (see production_net.log)"}
        try:
            out_r = out.reshape(reference.shape)
        except ValueError:
            return {"passed": False, "detail": f"shape mismatch ncnn {out.shape} vs ref {reference.shape}"}
        diff = np.abs(out_r.astype(np.float32) - np.asarray(reference, dtype=np.float32))
        passed = bool(np.allclose(out_r, reference, atol=tol, rtol=tol))
        return {"passed": passed,
                "max_diff": float(diff.max()), "mean_diff": float(diff.mean()),
                "detail": f"max_diff={float(diff.max()):.6f} out_name={out_name} in_names={in_names}"}

    # --- 3. benchmark (MoKA-style android+adb; skipped if no device) ------
    @property
    def android_build_dir(self) -> Path:
        return self.ncnn_root / "build-android-aarch64"

    def benchmark(self, model_param_path: str | Path, input_shapes_str: str,
                  retarget_to: str | None = None,
                  expected_src_type: str | None = None) -> dict:
        """Benchmark the converted model on-device via benchncnn.

        `retarget_to` (= the candidate's Cand_<Op> class) re-points the model's
        output-producing layer to OUR implementation before pushing, so benchncnn
        measures the kernel we generated rather than ncnn's built-in. Needed for
        ops ncnn already supports (the native conversion emits the built-in type);
        idempotent for new ops (the conversion already targets Cand_<Op>).
        """
        if not self.do_benchmark:
            return {"ran": False, "skipped": True, "reason": "benchmark disabled"}
        if not self.android_ndk:
            return {"ran": False, "skipped": True, "reason": "ANDROID_NDK env not set"}
        have, why = detect_android_device()
        if not have:
            return {"ran": False, "skipped": True, "reason": why}

        # re-point the output layer to our impl so benchncnn runs OURS, not built-in
        param_to_push = Path(model_param_path)
        retarget_n = 0
        if retarget_to:
            try:
                text = param_to_push.read_text(encoding="utf-8")
                new_text, retarget_n = retarget_param_output_layer(text, retarget_to,
                                                                   expected_src_type)
                rp = self.workdir / "model_retargeted.param"
                rp.write_text(new_text, encoding="utf-8")
                param_to_push = rp
            except Exception as exc:  # noqa: BLE001
                return {"ran": False, "skipped": True,
                        "reason": f"param retarget failed: {exc}"}

        log = self.workdir / "benchmark.log"
        self.android_build_dir.mkdir(parents=True, exist_ok=True)
        cmd_cfg = [
            "cmake",
            f"-DCMAKE_TOOLCHAIN_FILE={self.android_ndk}/build/cmake/android.toolchain.cmake",
            "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-21",
            "-DNCNN_VULKAN=OFF", "-DCMAKE_BUILD_TYPE=Release",
            "-DNCNN_BUILD_BENCHMARK=ON",
            str(self.ncnn_root),
        ]
        try:
            with log.open("w", encoding="utf-8") as f:
                f.write("$ " + " ".join(cmd_cfg) + "\n")
                cfg = subprocess.run(cmd_cfg, cwd=self.android_build_dir,
                                     stdout=f, stderr=subprocess.STDOUT, text=True)
                if cfg.returncode != 0:
                    return {"ran": False, "skipped": True, "reason": "android cmake configure failed; see benchmark.log"}
                bld_cmd = ["cmake", "--build", ".", "-j", str(self.build_jobs)]
                rc_bld, out_bld = _run_cmake_bounded(bld_cmd, timeout=1200,
                                                      cwd=self.android_build_dir)
                f.write("$ " + " ".join(bld_cmd) + "\n" + out_bld)
                if rc_bld != 0:
                    return {"ran": False, "skipped": True, "reason": "android build failed; see benchmark.log"}
        except Exception as exc:  # noqa: BLE001
            return {"ran": False, "skipped": True, "reason": f"android build crashed: {exc}"}

        benchncnn = self.android_build_dir / "benchmark" / "benchncnn"
        if not benchncnn.exists():
            return {"ran": False, "skipped": True, "reason": f"benchncnn not built at {benchncnn}"}

        try:
            subprocess.run(["adb", "shell", "mkdir", "-p", "/data/local/tmp/ncnn"], check=True, capture_output=True, timeout=10)
            subprocess.run(["adb", "push", str(benchncnn), "/data/local/tmp/ncnn/benchncnn"],
                           check=True, capture_output=True, timeout=30)
            subprocess.run(["adb", "push", str(param_to_push), "/data/local/tmp/ncnn/model.param"],
                           check=True, capture_output=True, timeout=30)
            subprocess.run(["adb", "shell", "chmod", "+x", "/data/local/tmp/ncnn/benchncnn"],
                           check=True, capture_output=True, timeout=10)
            test_cmd = (f"cd /data/local/tmp/ncnn && ./benchncnn 100 4 2 -1 1 "
                        f"param=model.param shape={input_shapes_str}")
            res = subprocess.run(["adb", "shell", test_cmd], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as e:
            return {"ran": False, "skipped": True, "reason": f"adb/benchmark timed out: {e.cmd}"}
        except subprocess.CalledProcessError as e:
            return {"ran": False, "skipped": True, "reason": f"adb step failed: {e.stderr or e}"}

        with log.open("a", encoding="utf-8") as f:
            f.write("\n" + res.stdout + "\n" + res.stderr)
        if res.returncode != 0:
            return {"ran": True, "skipped": False, "reason": "benchncnn crashed",
                    "retargeted": retarget_n, **parse_benchmark_output(res.stdout + res.stderr)}
        parsed = parse_benchmark_output(res.stdout)
        return {"ran": True, "skipped": False, "reason": "", "retargeted": retarget_n, **parsed}


def torch_input_shapes_str(model_py: str | Path) -> str:
    """Build the `shape=` arg for benchncnn: [w,h,c],[..] per input."""
    spec = importlib.util.spec_from_file_location("ds_model", str(model_py))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    shapes = []
    for t in mod.get_inputs():
        s = list(t.shape)
        if len(s) == 4:
            shapes.append(f"[{s[3]},{s[2]},{s[1]}]")
        elif len(s) == 3:
            shapes.append(f"[{s[2]},1,{s[1]}]")
        elif len(s) == 2:
            shapes.append(f"[{s[1]},{s[0]}]")
        else:
            shapes.append("[" + ",".join(str(x) for x in s) + "]")
    return ",".join(shapes)
