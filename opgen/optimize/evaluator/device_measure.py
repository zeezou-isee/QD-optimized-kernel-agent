"""On-device latency measurement for the OptimizeAgent evaluator.

The host CpuRunner times a candidate as macOS subprocess wall-clock (+fork/exec),
which is NOT the real phone runtime. For optimization to use latency as its
objective, candidate AND baseline latency must both be real on-phone time.

This wraps the already-built device oracles (`DeviceOracle` for base/arm,
`VulkanDeviceOracle` for vulkan) — which cross-compile the single-layer runner and
time it on the phone via `--bench` (clean min single-forward ms). Latency-only
(`measure_speedup=False`); host LayerOracle still does the correctness gate.

Availability is checked once (adb/NDK/lib); when no device, `latency()` returns
None and the caller falls back to the host harness (unchanged behavior).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class DeviceMeasurer:
    def __init__(self, backend: str, ncnn_root: str | Path | None = None, bench: int = 20) -> None:
        self.backend = backend
        self.bench = bench
        from layer_oracle import DeviceOracle, VulkanDeviceOracle
        self.oracle = (VulkanDeviceOracle(ncnn_root=ncnn_root) if backend == "vulkan"
                       else DeviceOracle(ncnn_root=ncnn_root))
        self._avail: bool | None = None
        self._why: str = ""

    def available(self) -> bool:
        if self._avail is None:
            self._avail, self._why = self.oracle.available()
        return self._avail

    def latency(self, *, candidate_cpp: str | Path, class_name: str, header: str,
                params: dict[int, Any] | None, inputs: Sequence[np.ndarray],
                reference: np.ndarray, weights: Sequence[np.ndarray] = (),
                weight_flags: Sequence[int] = (), extra_sources: Sequence[str | Path] = (),
                extra_includes: Sequence[str | Path] = (), packing: int = 0,
                shader: str | Path | None = None) -> float | None:
        """Min single-forward latency (ms) on the phone, or None if no device / the
        device run didn't produce a bench number (caller falls back to host)."""
        if not self.available():
            return None
        kw: dict[str, Any] = {"shader": shader} if self.backend == "vulkan" else {"packing": packing}
        try:
            r = self.oracle.verify(
                candidate_cpp=candidate_cpp, class_name=class_name, header=header,
                params=params, inputs=inputs, reference=reference, weights=weights,
                weight_flags=weight_flags, extra_sources=extra_sources,
                extra_includes=extra_includes, bench=self.bench, measure_speedup=False,
                backend=self.backend, **kw)
        except Exception:  # noqa: BLE001 — never break the search on a device hiccup
            return None
        return getattr(r, "latency", None)
