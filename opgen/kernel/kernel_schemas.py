"""Data contracts for the KernelAgent (from-scratch ncnn layer kernel writer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KernelProfile:
    """What the analyzer decided about the kernel to write."""

    task_name: str
    class_name: str = ""           # unique, must not collide with libncnn (e.g. Cand_Abs)
    header: str = ""               # e.g. cand_abs.h
    file: str = ""                 # e.g. cand_abs.cpp
    one_blob_only: bool = True
    support_inplace: bool = False
    params: dict[str, Any] = field(default_factory=dict)   # {param_id(str): value}
    weight_keys: list[str] = field(default_factory=list)   # state_dict keys, in load_model order
    analog_layer: str = ""         # nearest existing ncnn base layer to imitate (file stem)
    backend: str = "base"          # "base" | "arm" | "vulkan"
    base_class: str = ""           # for arm/vulkan: the base layer class it subclasses (Cand_Abs)
    shader: str = ""               # vulkan only: the .comp shader file (e.g. cand_abs.comp)
    notes: str = ""

    # naming suffix per backend (base has none); vulkan reserved for a later phase.
    _SUFFIX = {"base": "", "arm": "_arm", "vulkan": "_vulkan"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_llm(cls, task_name: str, payload: dict[str, Any],
                 backend: str = "base") -> "KernelProfile":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: v for k, v in (payload or {}).items() if k in allowed}
        clean["task_name"] = task_name
        clean["backend"] = backend
        prof = cls(**clean)
        # backend-aware default naming: base -> Cand_Abs / cand_abs.{h,cpp}
        #                                arm  -> Cand_Abs_arm / cand_abs_arm.{h,cpp}
        safe = "".join(ch if ch.isalnum() else "_" for ch in task_name)
        suffix = cls._SUFFIX.get(backend, "")
        base_class = f"Cand_{safe}"
        base_stem = f"cand_{safe.lower()}"
        if not prof.base_class:
            prof.base_class = base_class
        if not prof.class_name:
            prof.class_name = base_class + suffix
        if not prof.header:
            prof.header = f"{base_stem}{suffix}.h"
        if not prof.file:
            prof.file = f"{base_stem}{suffix}.cpp"
        # normalize params keys to str
        prof.params = {str(k): v for k, v in (prof.params or {}).items()}
        return prof

    def as_backend(self, backend: str) -> "KernelProfile":
        """Derive a same-op profile for another backend (e.g. base -> arm),
        reusing params/weights/flags/analog and re-deriving the names. Avoids a
        second analyzer LLM call — the arm layer inherits everything from base."""
        import copy
        p = copy.deepcopy(self)
        p.backend = backend
        base_class = self.base_class or self.class_name   # if self is base, class_name IS the base
        safe = base_class[len("Cand_"):] if base_class.startswith("Cand_") else base_class
        suffix = self._SUFFIX.get(backend, "")
        p.base_class = base_class
        p.class_name = base_class + suffix
        p.header = f"cand_{safe.lower()}{suffix}.h"
        p.file = f"cand_{safe.lower()}{suffix}.cpp"
        # vulkan authors a separate compute shader compiled at runtime
        p.shader = f"cand_{safe.lower()}.comp" if backend == "vulkan" else ""
        return p


@dataclass
class KernelResult:
    task_name: str
    profile: dict[str, Any] = field(default_factory=dict)

    identify_ok: bool = False
    generate_ok: bool = False
    compile_ok: bool = False
    numeric_ok: bool = False
    numeric_skipped: bool = False

    response_code: dict[str, str] = field(default_factory=dict)
    compile_error: str = ""
    numeric_log: str = ""
    max_diff: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.compile_ok and (self.numeric_ok or self.numeric_skipped)

    def first_failure(self) -> str | None:
        if not self.generate_ok:
            return "generate_repair"
        if not self.compile_ok:
            return "compile_repair"
        if not (self.numeric_ok or self.numeric_skipped):
            return "numeric_repair"
        return None

    def feedback(self, phase: str) -> str:
        return {
            "generate_repair": "No valid .h/.cpp code blocks were extracted.",
            "compile_repair": self.compile_error,
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
        d["numeric"] = self.numeric_status
        return d


@dataclass
class KernelRound:
    round_idx: int
    phase: str
    prompt_path: str
    response_path: str
    result_path: str
    ok: bool
    stages: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
