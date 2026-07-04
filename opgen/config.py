"""Central path/runtime configuration for graph_agent.

Everything is resolved relative to the repository layout:

    kernelgen/
      ncnn/                         <- the ncnn source tree we inject into
        tools/pnnx/                 <- PNNX converter (built from source)
      EndtoEndMobilekernelAgent/    <- this package
      MobileKernelBench_git/        <- datasets (PyTorch reference models)

The defaults can be overridden via environment variables or by constructing a
``GraphConfig`` explicitly, so the agent stays independent and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Anchor: this file lives in kernelgen/EndtoEndMobilekernelAgent/opgen/config.py
# ---------------------------------------------------------------------------
AGENT_ROOT = Path(__file__).resolve().parent          # .../EndtoEndMobilekernelAgent/opgen


def _find_kernelgen(start: Path) -> Path:
    """Walk up until we find the repo root (the dir containing an `ncnn/`)."""
    for p in [start, *start.parents]:
        if (p / "ncnn").is_dir():
            return p
    return start.parents[2]


KERNELGEN_ROOT = _find_kernelgen(AGENT_ROOT)           # .../kernelgen
# Single, tidy runtime root for ALL agents: runs/<task>/{kernel,graph,operator}
RUNS_ROOT = AGENT_ROOT / "runs"


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c and c.exists():
            return c
    return None


@dataclass
class GraphConfig:
    """Resolved paths and runtime knobs for one GraphAgent session."""

    # --- ncnn / pnnx source tree -------------------------------------------
    ncnn_root: Path = field(default_factory=lambda: KERNELGEN_ROOT / "ncnn")

    # --- LLM ---------------------------------------------------------------
    model: str = "deepseek-v4-pro"
    max_rounds: int = 8

    # --- behaviour ---------------------------------------------------------
    run_numeric: bool = True            # run end-to-end allclose (needs kernel)
    keep_changes_on_success: bool = False
    # If True, stop early when the current pnnx already converts the op correctly.
    # Default False: the agent's job is to author from scratch, verified against
    # PyTorch (not against any baseline conversion).
    skip_if_supported: bool = False
    build_jobs: int = 8
    # Optional libtorch install dir (passed as -DTorch_INSTALL_DIR).
    # If None, pnnx auto-probes the pip-installed PyTorch (PNNXProbeForPyTorchInstall).
    torch_install_dir: Path | None = None

    # --- vulkan-specific ---------------------------------------------------
    # How the vulkan KernelAgent uses ncnn's built-in <Op>_vulkan classes.
    #   "scratch"      → agent writes .h+.cpp+.comp from scratch (default;
    #                    the ncnn baseline is only consulted for pnnx-driven
    #                    profile calibration, never subclassed)
    #   "native_first" → try native-subclass first, fall back to from-scratch
    #                    if it doesn't verify
    #   "native_only"  → native-subclass only (fail if unavailable / doesn't
    #                    verify). Legacy path from the miniset/subset audit.
    vulkan_mode: str = "scratch"
    # --- dataset (PyTorch reference models) --------------------------------
    dataset_root: Path | None = None

    # --- outputs -----------------------------------------------------------
    run_root: Path = field(default_factory=lambda: RUNS_ROOT)

    def __post_init__(self) -> None:
        self.ncnn_root = Path(self.ncnn_root)
        self.run_root = Path(self.run_root)
        if self.torch_install_dir:
            self.torch_install_dir = Path(self.torch_install_dir)
        if self.dataset_root is None:
            self.dataset_root = _first_existing(
                KERNELGEN_ROOT / "MobileKernelBench_git" / "dataset" / "Mobilekernelbench",
                KERNELGEN_ROOT / "dataset" / "Mobilekernelbench",
            )
        else:
            self.dataset_root = Path(self.dataset_root)

    # --- derived pnnx paths ------------------------------------------------
    @property
    def pnnx_dir(self) -> Path:
        return self.ncnn_root / "tools" / "pnnx"

    @property
    def pnnx_src(self) -> Path:
        return self.pnnx_dir / "src"

    @property
    def pnnx_build(self) -> Path:
        return self.pnnx_dir / "build"

    @property
    def pnnx_bin(self) -> Path:
        return self.pnnx_build / "src" / "pnnx"

    @property
    def pass_ncnn_dir(self) -> Path:
        return self.pnnx_src / "pass_ncnn"

    @property
    def pass_level1_dir(self) -> Path:
        return self.pnnx_src / "pass_level1"

    @property
    def pass_level2_dir(self) -> Path:
        return self.pnnx_src / "pass_level2"

    @property
    def src_cmake(self) -> Path:
        return self.pnnx_src / "CMakeLists.txt"

    @property
    def tests_ncnn_dir(self) -> Path:
        return self.pnnx_dir / "tests" / "ncnn"

    @property
    def tests_ncnn_cmake(self) -> Path:
        return self.tests_ncnn_dir / "CMakeLists.txt"

    # The set(...) variable that each pass kind is registered under in src/CMakeLists.txt
    CMAKE_SRC_VAR = {
        "pass_ncnn": "pnnx_pass_ncnn_SRCS",
        "pass_level1": "pnnx_pass_level1_SRCS",
        "pass_level2": "pnnx_pass_level2_SRCS",
    }

    def run_dir(self, task_name: str) -> Path:
        # GraphAgent output. Route through paths.py so the layout has ONE
        # source of truth (see opgen/paths.py — new 5-stage runs/<task>/
        # layout). Current implementation returns runs/<task>/graph, but that
        # name is defined in paths.graph_dir; do not hardcode it here.
        import paths  # local import to avoid circular at module load time
        return paths.graph_dir(self.run_root, task_name)


# A module-level default, convenient for quick scripts / tests.
DEFAULT = GraphConfig()
