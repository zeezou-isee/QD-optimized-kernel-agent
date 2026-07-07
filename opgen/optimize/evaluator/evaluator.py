"""Evaluator — the truth gate facade (Workflow §5).

Per (template, point):
    materialize  → compile (LayerOracle/CpuRunner)
                 → correctness oracle (对拍 baseline)
                 → measure harness (warmup + N runs + noise floor)
                 → MeasureSample

The Evaluator owns everything needed to turn a *parameterized template* + a
*concrete parameter point* into a trustworthy `MeasureSample`. It derives the
kernel's inputs/params/weights from the PyTorch reference model once, compiles
and runs the **baseline** kernel once to obtain the correctness reference, and
thereafter every candidate is gated against that reference before timing.

Backend: CPU/ARM only at M1 (LayerOracle compiles candidate .cpp + libncnn.a).
Vulkan is left as a future backend (not wired here).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from layer_oracle import LayerOracle, VulkanLayerOracle, torch_to_ncnn_input
from schemas import (
    CorrectnessReport,
    MeasureSample,
    ParameterizedTemplate,
    materialize,
)

from .cpu_runner import CpuRunner
from .vk_runner import VkRunner
from .correctness_oracle import CorrectnessOracle
from .measure_harness import MeasureHarness


def _shader_file(code: dict[str, str]) -> str | None:
    """The vulkan candidate's .comp shader basename (vulkan kernels are a triple)."""
    return next((n for n in code if n.endswith(".comp")), None)


def _detect_class_name(code: dict[str, str]) -> str:
    # match `class X : public Layer` (base) OR `class X_arm : public X` (arm subclass)
    for _name, src in code.items():
        m = re.search(r"class\s+(\w+)\s*:\s*public\s+\w+", src or "")
        if m:
            return m.group(1)
    return ""


def _split_files(code: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (cpp_basename, h_basename) from a {basename: code} dict."""
    cpp = next((n for n in code if n.endswith((".cpp", ".cc", ".cxx"))), None)
    hdr = next((n for n in code if n.endswith((".h", ".hpp"))), None)
    return cpp, hdr


class Evaluator:
    """Truth gate: (template, point) -> MeasureSample.

    Parameters
    ----------
    baseline_kernel : {basename: code}  — already-verified base kernel (对拍参考)
    model_py        : PyTorch reference model file (provides inputs/weights)
    ncnn_root       : ncnn source tree with a prebuilt build_lib/ (libncnn.a)
    class_name/header/file : kernel identity; auto-detected from baseline if None
    weight_keys     : state_dict keys in load order (empty for unary ops)
    params          : fixed ncnn layer params {id: value} (empty for unary ops)
    """

    def __init__(
        self,
        *,
        baseline_kernel: dict[str, str],
        model_py: str | Path,
        ncnn_root: str | Path | None = None,
        workdir: str | Path | None = None,
        class_name: str | None = None,
        header: str | None = None,
        file: str | None = None,
        weight_keys: Sequence[str] | None = None,
        params: dict[int, Any] | None = None,
        warmup: int = 3,
        runs: int = 20,
        aggregate: str = "median",
        tol: float = 2e-3,
        backend: str = "base",
        base_files: dict[str, str] | None = None,   # arm: verified base layer code
        device_measure: bool = False,   # measure candidate latency on the REAL phone
        device_bench: int = 20,         # runner --bench iterations for device latency
    ) -> None:
        self.baseline_kernel = dict(baseline_kernel)
        self.model_py = str(model_py)
        self.backend = backend
        self._ncnn_root = ncnn_root
        cpp, hdr = _split_files(self.baseline_kernel)
        self.class_name = class_name or _detect_class_name(self.baseline_kernel)
        self.header = header or hdr or ""
        self.file = file or cpp or ""
        if not (self.class_name and self.header and self.file):
            raise ValueError(
                f"could not resolve kernel identity (class={self.class_name!r} "
                f"header={self.header!r} file={self.file!r}); pass them explicitly")
        self.weight_keys = list(weight_keys or [])
        self.params = {int(k): v for k, v in (params or {}).items()}
        self.tol = tol

        # vulkan optimizes on the GPU (VulkanLayerOracle, runtime-compiled shader);
        # base/arm use the CPU LayerOracle.
        if backend == "vulkan":
            self.oracle = VulkanLayerOracle()
            self.runner = VkRunner(self.oracle)
        else:
            self.oracle = LayerOracle(ncnn_root=ncnn_root, workdir=workdir)
            self.runner = CpuRunner(self.oracle)
        self.harness = MeasureHarness(self.runner, warmup=warmup, runs=runs,
                                      aggregate=aggregate)
        self._cand_dir = self.oracle.workdir / "_cand"
        self._cand_dir.mkdir(parents=True, exist_ok=True)

        # arm/vulkan subclass the verified base layer -> drop the base files next to
        # the candidate and compile the base .cpp in as an extra source.
        self.extra_sources: list[Path] = []
        self.extra_includes: list[Path] = []
        self.packing = 0
        # vulkan: the candidate's .comp shader (compiled at runtime); injected per run.
        self.shader_name = _shader_file(self.baseline_kernel) if backend == "vulkan" else None
        if backend in ("arm", "vulkan"):
            for name, content in (base_files or {}).items():
                p = self._cand_dir / name
                p.write_text(content, encoding="utf-8")
                if name.endswith((".cpp", ".cc", ".cxx")):
                    self.extra_sources.append(p)
        if backend == "arm":
            self.extra_includes = [self.oracle.ncnn_root / "src" / "layer" / "arm"]
            self.packing = 4

        # derive I/O once, then compile+run the baseline to fix the reference.
        self.inputs, self.weights = self._derive_io()
        self._ref = self._baseline_reference()   # reuse for correctness + device compare
        self.correctness = CorrectnessOracle(
            self._ref, atol=tol, rtol=tol, backend=backend,
            input=(self.inputs[0] if self.inputs else None))

        # on-device latency (real phone). When enabled + a device is present, the
        # candidate's latency comes from the phone (--bench) instead of host
        # wall-clock — so baseline AND candidates are same-harness real-phone time
        # and the search optimizes actual device latency. Falls back to host when
        # no device. Correctness stays on the host LayerOracle (fast).
        self.device_measure = bool(device_measure)
        self._measurer = None
        if self.device_measure:
            from .device_measure import DeviceMeasurer
            self._measurer = DeviceMeasurer(backend, ncnn_root=self._ncnn_root, bench=device_bench)

    # ------------------------------------------------------------------ I/O
    def _derive_io(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        import torch
        spec = importlib.util.spec_from_file_location("ds_model_opt", self.model_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        model = (mod.Model(*init) if init else mod.Model()).eval()
        inputs = mod.get_inputs()
        ncnn_inputs = [torch_to_ncnn_input(t.detach().numpy()) for t in inputs]
        sd = model.state_dict()
        weights = []
        for k in self.weight_keys:
            if k not in sd:
                raise KeyError(f"weight key {k!r} not in state_dict {list(sd)}")
            weights.append(sd[k].detach().numpy().reshape(-1))
        return ncnn_inputs, weights

    # ----------------------------------------------------------- compile+run
    def _shader_path(self) -> Path | None:
        """vulkan: the .comp written into _cand_dir for the current candidate."""
        return (self._cand_dir / self.shader_name) if self.shader_name else None

    def _compile_run_once(self, code: dict[str, str]) -> tuple[bool, np.ndarray | None, Any, str]:
        """Write code, compile, run once. Returns (ok, output|None, art|None, err)."""
        cpp, hdr = _split_files(code)
        if cpp is None:
            return False, None, None, "no .cpp among kernel files"
        for name, src in code.items():
            (self._cand_dir / name).write_text(src, encoding="utf-8")
        cpp_path = self._cand_dir / cpp
        try:
            art, _clog = self.runner.compile_only(
                candidate_cpp=cpp_path, class_name=self.class_name,
                header=hdr or self.header, inputs=self.inputs,
                weights=self.weights, params=self.params or None,
                extra_sources=self.extra_sources, extra_includes=self.extra_includes,
                packing=self.packing, shader=self._shader_path())
        except Exception as exc:  # noqa: BLE001 — compile failure
            return False, None, None, f"compile failed: {exc}"
        ok, _ms, err = self.runner.run_once(art)
        if not ok:
            return False, None, art, f"runtime crash: {err}"
        return True, self.runner.read_output(art), art, ""

    def _baseline_reference(self) -> np.ndarray:
        ok, out, _art, err = self._compile_run_once(self.baseline_kernel)
        if not ok or out is None:
            raise RuntimeError(f"baseline kernel did not produce a reference: {err}")
        return out

    # -------------------------------------------------------------- evaluate
    @staticmethod
    def _precision_hints(template: ParameterizedTemplate) -> tuple[bool, bool]:
        """Derive (fp16_storage, fp16_arith) from template.techniques (P2 opt-in).
        The LLM proposer declares tags like "fp16-storage" / "fp16" / "fp16-arith"
        when it produces a half-precision variant; we forward them to the runner
        so ncnn's fp16 code paths get enabled. arith implies storage."""
        tags = " ".join(t.lower() for t in (template.techniques or []))
        arith = ("fp16-arith" in tags) or ("fp16_arith" in tags) or ("fp16-arithmetic" in tags)
        storage = arith or ("fp16-storage" in tags) or ("fp16_storage" in tags) or \
                  ("fp16-packed" in tags) or (" fp16" in f" {tags}") or tags.startswith("fp16")
        return storage, arith

    def evaluate(self, template: ParameterizedTemplate, point: dict[str, Any]) -> MeasureSample:
        """Materialize → compile → correctness → measure. Never raises on a bad
        candidate; failures come back as a MeasureSample with correct=False."""
        try:
            code = materialize(template, point)
        except KeyError as exc:
            return MeasureSample(point=point, correct=False, error=f"materialize: {exc}")

        cpp, hdr = _split_files(code)
        if cpp is None:
            return MeasureSample(point=point, correct=False, error="no .cpp among kernel files")
        for name, src in code.items():
            (self._cand_dir / name).write_text(src, encoding="utf-8")
        cpp_path = self._cand_dir / cpp

        fp16_storage, fp16_arith = self._precision_hints(template)

        # compile
        try:
            art, clog = self.runner.compile_only(
                candidate_cpp=cpp_path, class_name=self.class_name,
                header=hdr or self.header, inputs=self.inputs,
                weights=self.weights, params=self.params or None,
                extra_sources=self.extra_sources, extra_includes=self.extra_includes,
                packing=self.packing, fp16_storage=fp16_storage,
                fp16_arith=fp16_arith, shader=self._shader_path())
        except Exception as exc:  # noqa: BLE001
            tail = str(exc)[-400:]
            return MeasureSample(point=point, correct=False,
                                 error="compile failed", compile_log_tail=tail)

        # one run for correctness
        ok, _ms, err = self.runner.run_once(art)
        if not ok:
            return MeasureSample(point=point, correct=False, error=f"runtime crash: {err}")
        report = self.correctness.check(self.runner.read_output(art))
        if not report.passed:
            return MeasureSample(point=point, correct=False, correctness=report,
                                 error="incorrect output")

        # timed measurement — REAL PHONE when device_measure + a device is present
        # (candidate + baseline both go through this path -> consistent real-phone
        # latency), else host wall-clock (fallback / no-device / CI).
        if self._measurer is not None:
            dev_lat = self._measurer.latency(
                candidate_cpp=cpp_path, class_name=self.class_name,
                header=hdr or self.header, params=self.params or None,
                inputs=self.inputs, reference=self._ref, weights=self.weights,
                extra_sources=self.extra_sources, extra_includes=self.extra_includes,
                packing=self.packing, shader=self._shader_path())
            if dev_lat is not None:
                return MeasureSample(
                    point=point, correct=True, latency_ms=dev_lat, latency_min_ms=dev_lat,
                    latency_median_ms=dev_lat, latency_std_ms=0.0,
                    n_runs=self._measurer.bench, correctness=report)

        try:
            stats = self.harness.measure(art)
        except RuntimeError as exc:
            return MeasureSample(point=point, correct=False, correctness=report,
                                 error=f"measure failed: {exc}")
        return MeasureSample(
            point=point, correct=True,
            latency_ms=stats.latency_ms, latency_min_ms=stats.min_ms,
            latency_median_ms=stats.median_ms, latency_std_ms=stats.std_ms,
            n_runs=stats.n_runs, correctness=report,
        )
