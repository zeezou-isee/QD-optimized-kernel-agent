"""Single source of truth for `opgen/runs/<task>/` layout.

Every module that reads or writes a subdir under `runs/<task>/` must go through
this file — no hardcoded string literals like "kernel" / f"kernel_{backend}"
elsewhere. Adding a new backend (cuda, metal, ...) is then zero code change:
you just create `runs/<task>/backends/<backend>/kernel/` and it works.

Target layout (5-stage):
    runs/<task>/
    ├─ analyze/              # backend-invariant analysis
    │  ├─ introspect.json    # shape / weights / init_inputs (from PyTorch)
    │  ├─ kernel_profile.json  # analog_layer / params / weight_keys (invariant fields)
    │  └─ pnnx_probe/        # .ncnn.param + .ncnn.bin + _ncnn.py
    ├─ base_kernel/          # from-scratch base kernel (single source of truth)
    │  ├─ round_*, summary.json, history.json, memory.json, config.json
    │  └─ artifacts/         # CONTRACT dir: cand_<op>.h/.cpp + kernel_profile.json
    ├─ graph/                # GraphAgent output (already shared)
    ├─ operator/             # OperatorAgent private state
    ├─ adapter/              # AdapterAgent private state
    └─ backends/
       ├─ base/optimize/         # base kernel's optimization run
       ├─ arm/{kernel,optimize}/ # arm kernel + its optimization
       ├─ vulkan/{kernel,optimize}/
       └─ <future>/{kernel,optimize}/

Legacy layout (pre-restructure) — READ fallback still works via kernel_summary():
    runs/<task>/kernel/       # base
    runs/<task>/kernel_arm/
    runs/<task>/kernel_vulkan/
    runs/<task>/optimize/     # assumed base
"""

from __future__ import annotations

from pathlib import Path


# ----------------------------------------------------------------- roots
def task_root(runs_root: Path | str, task: str) -> Path:
    return Path(runs_root) / task


# ------------------------------------------------------ backend-invariant
def analyze_dir(runs_root: Path | str, task: str) -> Path:
    return task_root(runs_root, task) / "analyze"


def introspect_json(runs_root: Path | str, task: str) -> Path:
    return analyze_dir(runs_root, task) / "introspect.json"


def kernel_profile_shared_json(runs_root: Path | str, task: str) -> Path:
    """The backend-invariant subset of KernelProfile.
    Excludes: class_name, header, file, shader, backend, base_class,
    native_vulkan, native_vulkan_class, native_vulkan_header.
    """
    return analyze_dir(runs_root, task) / "kernel_profile.json"


def analyze_pnnx_probe_dir(runs_root: Path | str, task: str) -> Path:
    return analyze_dir(runs_root, task) / "pnnx_probe"


# --------------------------------------------------------- base kernel
def base_kernel_dir(runs_root: Path | str, task: str) -> Path:
    return task_root(runs_root, task) / "base_kernel"


def base_kernel_artifacts_dir(runs_root: Path | str, task: str) -> Path:
    """Contract: this dir holds ONLY the final verified base .h/.cpp files
    plus a kernel_profile.json. arm/vulkan/optimize readers pull from here.
    """
    return base_kernel_dir(runs_root, task) / "artifacts"


def base_kernel_summary(runs_root: Path | str, task: str) -> Path:
    return base_kernel_dir(runs_root, task) / "summary.json"


# --------------------------------------------------------- graph (unchanged)
def graph_dir(runs_root: Path | str, task: str) -> Path:
    return task_root(runs_root, task) / "graph"


# --------------------------------------------------------- per-backend
def backends_root(runs_root: Path | str, task: str) -> Path:
    return task_root(runs_root, task) / "backends"


def backend_kernel_dir(runs_root: Path | str, task: str, backend: str) -> Path:
    """Per-backend from-scratch kernel dir.

    Convention: `backend` here is "arm" / "vulkan" / etc — NOT "base". The
    from-scratch base kernel lives at `base_kernel_dir(...)`, NOT
    `backends/base/kernel/`. That's because base is treated as the SoT and
    every other backend subclasses it; a "backends/base/kernel/" would confuse
    "the base kernel" vs "the base backend's optimize starting point".
    """
    return backends_root(runs_root, task) / backend / "kernel"


def backend_optimize_dir(runs_root: Path | str, task: str, backend: str) -> Path:
    """Per-backend optimize output. `backend` can be "base" here — that's the
    optimize phase run on the base kernel. arm/vulkan optimize likewise live
    under backends/<backend>/optimize/."""
    return backends_root(runs_root, task) / backend / "optimize"


# --------------------------------------------------------- lookup helpers
def kernel_summary(runs_root: Path | str, task: str, backend: str) -> Path:
    """Resolve the kernel summary.json for (task, backend).

    Returns the NEW-layout path if it exists, else the LEGACY path if IT
    exists, else the new path (missing — caller should check .exists()).
    This lets callers keep working during the migration period.
    """
    runs_root = Path(runs_root)
    if backend == "base":
        new = base_kernel_summary(runs_root, task)
    else:
        new = backend_kernel_dir(runs_root, task, backend) / "summary.json"
    if new.exists():
        return new
    legacy_sub = "kernel" if backend == "base" else f"kernel_{backend}"
    legacy = task_root(runs_root, task) / legacy_sub / "summary.json"
    if legacy.exists():
        return legacy
    return new  # neither exists — return the "correct" (missing) path


def legacy_kernel_dir(runs_root: Path | str, task: str, backend: str) -> Path:
    """Explicit legacy subdir for callers that need to check `.exists()`
    before falling back."""
    sub = "kernel" if backend == "base" else f"kernel_{backend}"
    return task_root(runs_root, task) / sub


# --------------------------------------------------------- profile stripping
# The set of KernelProfile fields that are backend-specific and should be
# STRIPPED before writing the shared kernel_profile.json under analyze/.
BACKEND_SPECIFIC_PROFILE_FIELDS: tuple[str, ...] = (
    "class_name", "header", "file", "shader", "backend", "base_class",
    "native_vulkan", "native_vulkan_class", "native_vulkan_header",
)


def strip_backend_fields(profile_dict: dict) -> dict:
    """Return a copy of a KernelProfile dict with backend-specific fields
    removed. Used when saving the shared analyze/kernel_profile.json."""
    return {k: v for k, v in profile_dict.items()
            if k not in BACKEND_SPECIFIC_PROFILE_FIELDS}
