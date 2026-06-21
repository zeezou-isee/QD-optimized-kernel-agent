"""HardwareSpecs — cheap, best-effort host CPU profile for analytic pruning.

Workflow §4.4 / 微观参数优化 §一: the constraint engine needs physical bounds
(cache sizes, SIMD width, register budget) to derive the feasible region before
any real measurement. We read what the OS exposes for free; everything has a
safe fallback so the pipeline never hard-fails on an exotic host.

M1 = CPU/ARM. `vector_bits` is 128 for NEON (Apple Silicon / ARMv8) and 256 for
x86 AVX2 hosts (so the same code path can be sanity-checked on a dev laptop).
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class HardwareSpecs:
    arch: str                 # "arm64" | "x86_64" | ...
    l1d_bytes: int            # per-core L1 data cache
    l2_bytes: int             # L2 cache
    vector_bits: int          # SIMD register width (NEON=128, AVX2=256)
    n_cores: int              # physical cores (best-effort)
    # number of registers usable for accumulators — a coarse spill guard knob.
    # ARMv8 has 32 NEON regs; x86_64 SSE/AVX has 16. Used only as a soft bound.
    vector_regs: int = 32

    @property
    def fp32_per_vector(self) -> int:
        return max(1, self.vector_bits // 32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fp32_per_vector"] = self.fp32_per_vector
        return d


def _sysctl_int(key: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def _linux_cache_bytes(index_kind: str) -> int | None:
    # /sys/devices/system/cpu/cpu0/cache/indexN/{level,type,size}
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        from pathlib import Path
        for idx in Path(base).glob("index*"):
            level = (idx / "level").read_text().strip()
            ctype = (idx / "type").read_text().strip().lower()
            size = (idx / "size").read_text().strip()  # e.g. "32K"
            want = ("1", "data") if index_kind == "l1d" else ("2", None)
            if level == want[0] and (want[1] is None or ctype == want[1]):
                mult = 1024 if size.endswith("K") else (1024 * 1024 if size.endswith("M") else 1)
                return int(size.rstrip("KMB") or 0) * mult
    except Exception:  # noqa: BLE001
        pass
    return None


def detect() -> HardwareSpecs:
    """Detect the host CPU profile (darwin/linux), with safe fallbacks."""
    arch = platform.machine().lower()
    system = platform.system().lower()
    is_arm = arch in ("arm64", "aarch64")
    vector_bits = 128 if is_arm else 256          # NEON vs AVX2
    vector_regs = 32 if is_arm else 16

    l1d = l2 = n_cores = None
    if system == "darwin":
        l1d = _sysctl_int("hw.perflevel0.l1dcachesize") or _sysctl_int("hw.l1dcachesize")
        l2 = _sysctl_int("hw.perflevel0.l2cachesize") or _sysctl_int("hw.l2cachesize")
        n_cores = _sysctl_int("hw.perflevel0.physicalcpu") or _sysctl_int("hw.physicalcpu")
    elif system == "linux":
        l1d = _linux_cache_bytes("l1d")
        l2 = _linux_cache_bytes("l2")

    import os
    return HardwareSpecs(
        arch=arch,
        l1d_bytes=l1d or 32 * 1024,
        l2_bytes=l2 or 256 * 1024,
        vector_bits=vector_bits,
        n_cores=n_cores or (os.cpu_count() or 4),
        vector_regs=vector_regs,
    )
