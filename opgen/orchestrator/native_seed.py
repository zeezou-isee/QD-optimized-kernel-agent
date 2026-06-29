"""native_seed — inject a framework's NATIVE operator implementation into the
experience pool (兵器谱) as a floor seed for the OptimizeAgent's MAP-Elites loop.

Why this exists
---------------
For an operator that ncnn already supports natively, the native .cpp/.h is a
high-quality reference kernel. We do NOT want it as the KernelAgent baseline
template (that would collapse QD diversity onto the one niche the native impl
already occupies — see AgentDesign/prologue/算子端到端优化全流程.md §3). Instead
the right place for it is the experience pool: dropped into the archive as an
initial elite, it occupies its own niche as a "floor" the search must beat, and
serves as a parent the LLM proposer can study — without constraining which other
niches get explored (experience_pool.py: "慢种子只要占空格子就存活").

This module reads the native source, infers its regime (roofline) and niche cell
(behavior descriptor), and appends a PoolRecord to the pool JSON. It reuses the
existing policy API verbatim — no new optimization machinery.
"""

from __future__ import annotations

from pathlib import Path

# Flat imports (bootstrap_paths put opgen/optimize on sys.path; policy re-exports).
from policy import (  # type: ignore
    ExperiencePool,
    PoolRecord,
    classify,
    diagnose,
    estimate_operator_profile,
)

# Source-level keyword -> behavior-descriptor technique tag. The tags are the
# same vocabulary bd.classify() matches on (substring, case-insensitive), so a
# hit here lands the seed in the right niche cell. Order doesn't matter; tags
# are unioned. Keep this conservative — a wrong tag mis-places the seed, but an
# absent tag only falls back to the most generic cell (still a valid floor).
_TAG_PATTERNS: tuple[tuple[str, str], ...] = (
    # layout / packing (memory-bound axis1)
    ("nc4hw4", "packed"), ("pack4", "packed"), ("elempack", "packed"),
    ("nhwc", "nhwc"),
    # tiling (memory-bound axis2)
    ("tile", "tiling"), ("blocking", "blocking"),
    # algo family (compute-bound axis1)
    ("im2col", "gemm"), ("sgemm", "gemm"), ("winograd", "winograd"),
    ("fft", "fft"), ("depthwise", "depthwise"),
    # compute mapping (compute-bound axis2)
    ("vdotq", "dotprod"), ("vdot_", "dotprod"), ("sdot", "dotprod"),
    ("vmlaq", "neon"), ("vld1q", "neon"), ("float32x4", "neon"),
    ("__arm_neon", "neon"), ("vfmaq", "neon"),
)


def _native_paths(ncnn_root: Path, ncnn_layer_type: str, backend: str) -> list[Path]:
    """Resolve native source paths for an ncnn layer type.

    ncnn convention: src/layer/<lower>.{cpp,h}; arm variant lives in
    src/layer/arm/<lower>_arm.{cpp,h}.
    """
    lower = ncnn_layer_type.lower()
    base = ncnn_root / "src" / "layer"
    paths = [base / f"{lower}.cpp", base / f"{lower}.h"]
    if backend == "arm":
        arm = base / "arm"
        paths += [arm / f"{lower}_arm.cpp", arm / f"{lower}_arm.h"]
    return paths


def _extract_tags(source_blob: str) -> list[str]:
    """Scan native source for technique markers (substring, case-insensitive)."""
    blob = source_blob.lower()
    tags: list[str] = []
    for needle, tag in _TAG_PATTERNS:
        if needle in blob and tag not in tags:
            tags.append(tag)
    return tags


def seed_native_into_pool(
    *,
    ncnn_root: str | Path,
    ncnn_layer_type: str,
    model_py: str | Path,
    pool_path: str | Path,
    backend: str = "base",
    measured_latency_ms: float | None = None,
    hardware: str = "",
) -> dict:
    """Read ncnn's native implementation of `ncnn_layer_type`, infer its regime +
    niche cell, and append it to the experience pool at `pool_path` as a floor seed.

    Returns a dict {seeded, regime, cell, files, note} suitable for a summary phase.
    Never raises on a missing/empty native source — returns seeded=False instead,
    so the caller can treat seeding as a non-blocking gain.

    Idempotent: if a record with the same (op_class, regime, cell) already exists
    in the pool, no duplicate is added.
    """
    ncnn_root = Path(ncnn_root)
    paths = _native_paths(ncnn_root, ncnn_layer_type, backend)
    kernel_code: dict[str, str] = {}
    for p in paths:
        if p.exists():
            kernel_code[p.name] = p.read_text(encoding="utf-8", errors="replace")
    if not kernel_code:
        return {"seeded": False, "regime": None, "cell": None, "files": [],
                "note": f"native source not found for '{ncnn_layer_type}' "
                        f"(looked in {ncnn_root / 'src' / 'layer'})"}

    # regime: roofline diagnosis from the PyTorch reference model.
    try:
        profile = estimate_operator_profile(str(model_py))
        regime = diagnose(profile).regime
    except Exception as exc:  # noqa: BLE001 — profiling shouldn't abort seeding
        return {"seeded": False, "regime": None, "cell": None,
                "files": list(kernel_code), "note": f"roofline profiling failed: {exc}"}

    # niche cell: behavior descriptor from source-level technique tags.
    tags = _extract_tags("\n".join(kernel_code.values()))
    cell = list(classify(tags, regime))

    # floor latency: inf by default so any real search result beats it in-cell;
    # the seed only occupies an empty niche + serves as a parent reference.
    latency_ms = float(measured_latency_ms) if measured_latency_ms is not None else float("inf")

    pool = ExperiencePool(pool_path)
    for r in pool.records:
        if r.op_class == ncnn_layer_type and r.regime == regime and list(r.cell) == cell:
            return {"seeded": False, "regime": regime, "cell": cell,
                    "files": list(kernel_code),
                    "note": "already seeded (idempotent skip)"}

    pool.records.append(PoolRecord(
        regime=regime, cell=cell, latency_ms=latency_ms, kernel_code=kernel_code,
        params={}, techniques=tags, op_class=ncnn_layer_type, hardware=hardware,
    ))
    pool.save()
    return {"seeded": True, "regime": regime, "cell": cell,
            "files": list(kernel_code), "techniques": tags,
            "note": f"native '{ncnn_layer_type}' seeded into pool ({backend})"}
