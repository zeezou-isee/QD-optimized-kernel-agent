"""Data contracts for graph_agent.

These mirror the spirit of MoKA's schemas but are specific to the ncnn graph
conversion problem (identify -> inject -> build -> convert -> verify).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


# ---------------------------------------------------------------------------
# Operator identification result
# ---------------------------------------------------------------------------
@dataclass
class OpProfile:
    """What the analyzer figured out about the operator to convert.

    Filled by the LLM analyzer (graph_prompts.analyzer_prompt) and validated /
    defaulted in code. Drives which pass files to write and how to verify.
    """

    task_name: str
    # nn_module | functional | aten | composite
    source_form: str = "functional"
    # unary | binary | weighted | tensor_manip | composite
    category: str = "unary"
    # target ncnn layer type string, e.g. "HardSigmoid" (used for structural check + type_str)
    target_ncnn_layer: str = ""
    # set by the analyzer (grounded on the pnnx probe): the op already converts to a
    # native ncnn layer (target is then that generic type, e.g. UnaryOp), so no new
    # layer/pass is needed. Informational; the orchestrator's existence check also detects this.
    already_supported: bool = False
    needs_weight: bool = False
    # torch entry point, e.g. "F.hardsigmoid" / "aten::hardsigmoid"
    torch_op: str = ""
    # ranks the test should cover, e.g. [1, 2, 3, 4]
    rank_coverage: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # files the agent intends to write (repo-relative), filled by analyzer/coder
    files_to_write: list[str] = field(default_factory=list)
    # most similar existing ops to imitate, e.g. ["F_hardsigmoid", "F_relu6"]
    analog_ops: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_llm(cls, task_name: str, payload: dict[str, Any]) -> "OpProfile":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in (payload or {}).items() if k in allowed}
        clean["task_name"] = task_name
        return cls(**clean)


# ---------------------------------------------------------------------------
# File injection bookkeeping (for clean restore)
# ---------------------------------------------------------------------------
@dataclass
class BackupHandle:
    """Records the source-tree mutations of one injection so they can be undone.

    Unlike MoKA (which only overwrites existing files), graph_agent CREATES new
    files and PATCHES CMakeLists, so restore must delete new files and revert
    patched ones.
    """

    created_files: list[str] = field(default_factory=list)        # delete on restore
    modified_files: dict[str, str] = field(default_factory=dict)  # path -> original text

    def to_dict(self) -> dict[str, Any]:
        return {"created_files": self.created_files, "modified_files": list(self.modified_files)}


# ---------------------------------------------------------------------------
# Pipeline result (normalized across all stages)
# ---------------------------------------------------------------------------
@dataclass
class GraphResult:
    task_name: str
    op_profile: dict[str, Any] = field(default_factory=dict)

    # stage flags (the 4-stage conversion pipeline)
    identify_ok: bool = False
    inject_ok: bool = False
    build_ok: bool = False
    convert_ok: bool = False
    structural_ok: bool = False
    numeric_ok: bool = False
    numeric_skipped: bool = False

    # generated artifacts
    response_code: dict[str, str] = field(default_factory=dict)   # repo-relative path -> code

    # diagnostics / feedback for the next round
    inject_error: str = ""
    build_error: str = ""
    convert_log: str = ""        # residual aten/prim, pnnx stdout warnings
    structural_log: str = ""     # what was/ wasn't found in .pnnx/.ncnn.param
    numeric_log: str = ""        # allclose diffs / shape mismatch

    artifacts: dict[str, str] = field(default_factory=dict)       # name -> path
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        numeric = self.numeric_ok or self.numeric_skipped
        return self.inject_ok and self.build_ok and self.convert_ok and self.structural_ok and numeric

    def first_failure(self) -> str | None:
        """Which stage to repair next (None == fully passed)."""
        if not self.inject_ok:
            return "inject_repair"
        if not self.build_ok:
            return "build_repair"
        if not self.convert_ok:
            return "convert_repair"
        if not self.structural_ok:
            return "convert_repair"
        if not (self.numeric_ok or self.numeric_skipped):
            return "numeric_repair"
        return None

    def feedback(self, phase: str) -> str:
        """The targeted diagnostic text to hand the debugger for ``phase``."""
        return {
            "inject_repair": self.inject_error,
            "build_repair": self.build_error,
            "convert_repair": (self.convert_log + "\n" + self.structural_log).strip(),
            "numeric_repair": self.numeric_log,
        }.get(phase, "")

    @property
    def numeric_status(self) -> str:
        """Human-readable numeric state: passed | failed | skipped."""
        if self.numeric_skipped:
            return "skipped"
        return "passed" if self.numeric_ok else "failed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        d["numeric"] = self.numeric_status  # "passed" | "failed" | "skipped" (clearer than numeric_ok)
        return d


@dataclass
class GraphRound:
    round_idx: int
    phase: str
    prompt_path: str
    response_path: str
    result_path: str
    ok: bool
    stages: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
