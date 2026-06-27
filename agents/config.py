"""Central path/runtime configuration for the agents package.

Everything is resolved relative to the repository layout:

    KernelAgent/                    <- repo root
      agents/                       <- this package
      frameworks/ncnn/              <- the ncnn source tree we inject into
        tools/pnnx/                 <- PNNX converter (built from source)
      datasets/MobileKernelBench/   <- PyTorch reference models

The defaults can be overridden via environment variables or by constructing a
``GraphConfig`` explicitly, so the agent stays independent and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Anchor: this file lives in KernelAgent/agents/config.py
# ---------------------------------------------------------------------------
AGENT_ROOT = Path(__file__).resolve().parent          # .../KernelAgent/agents


def _find_repo_root(start: Path) -> Path:
    """Walk up until we find the repo root (the dir containing `frameworks/ncnn`
    or a top-level `ncnn/`)."""
    for p in [start, *start.parents]:
        if (p / "frameworks" / "ncnn").is_dir() or (p / "ncnn").is_dir():
            return p
    return start.parents[1]


REPO_ROOT = _find_repo_root(AGENT_ROOT)                # .../KernelAgent


def _resolve_ncnn_root(repo: Path) -> Path:
    """Prefer frameworks/ncnn, fall back to a top-level ncnn/."""
    fw = repo / "frameworks" / "ncnn"
    return fw if fw.is_dir() else repo / "ncnn"


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
    ncnn_root: Path = field(default_factory=lambda: _resolve_ncnn_root(REPO_ROOT))

    # --- LLM ---------------------------------------------------------------
    model: str = "anthropic/claude-sonnet-4.5"
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
                REPO_ROOT / "datasets" / "MobileKernelBench",
                REPO_ROOT / "MobileKernelBench_git" / "dataset" / "Mobilekernelbench",
                REPO_ROOT / "dataset" / "Mobilekernelbench",
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
        # GraphAgent output: runs/<task>/graph
        return self.run_root / task_name / "graph"


# A module-level default, convenient for quick scripts / tests.
DEFAULT = GraphConfig()
