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
    # Functional ops (F.conv2d / F.linear / ...) carry weights as forward INPUTS,
    # not as nn.Parameter in state_dict. weights_from_inputs gives the 0-based
    # input indices that are actually weights (in load_model order). When this
    # is non-empty the verify pipeline pulls those inputs out and routes them to
    # the runner's `--weight` slot (mb.load), leaving the rest as bottom_blobs.
    weights_from_inputs: list[int] = field(default_factory=list)
    analog_layer: str = ""         # nearest existing ncnn base layer to imitate (file stem)
    backend: str = "base"          # "base" | "arm" | "vulkan"
    base_class: str = ""           # for arm/vulkan: the base layer class it subclasses (Cand_Abs)
    shader: str = ""               # vulkan only: the .comp shader file (e.g. cand_abs.comp)
    # vulkan only: when the op maps to a native ncnn <analog>_vulkan layer, the
    # candidate is a THIN SUBCLASS of it (Cand_X_vulkan : public <analog>_vulkan)
    # that inherits ncnn's verified create_pipeline + baked SPIR-V — no from-scratch
    # .comp needed. Set by KernelAgent when a native vulkan layer exists.
    native_vulkan: bool = False
    native_vulkan_class: str = ""  # e.g. "BinaryOp_vulkan"
    native_vulkan_header: str = "" # e.g. "vulkan/binaryop_vulkan.h"
    notes: str = ""

    # naming suffix per backend (base has none); vulkan reserved for a later phase.
    _SUFFIX = {"base": "", "arm": "_arm", "vulkan": "_vulkan"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Identifier-shaped values only — LLMs occasionally splat .h/.cpp source bodies
    # into these fields (we've seen it for And/Logic ops); a single bad value
    # cascades into 15 rounds of compile_repair because the field is plumbed into
    # `-DCANDIDATE_HEADER=<source body>` and `new <code>()` at the runner level.
    _FIELD_MAX_LEN = {"class_name": 80, "header": 80, "file": 80}
    # any of these characters in an identifier field → LLM put code there
    _FIELD_BAD_CHARS = ("\n", "\r", "#", "{", "}", ";", " ", "\t", '"', "'", "(", ")")

    @classmethod
    def _sanitize_identifier_field(cls, name: str, value: str) -> str:
        """Reject values that obviously aren't simple identifiers / filenames.

        Returns "" when the value is bad — caller then fills in the default
        (Cand_<Task> / cand_<task>.h / cand_<task>.cpp) so the kernel can
        still compile. Prints a clear warning so the bug is visible.
        """
        if not isinstance(value, str):
            return ""
        v = value.strip()
        if not v:
            return ""
        if len(v) > cls._FIELD_MAX_LEN[name]:
            print(f"[profile] WARNING: `{name}` looks like injected code "
                  f"({len(v)} chars, max {cls._FIELD_MAX_LEN[name]}); "
                  f"falling back to default")
            return ""
        if any(c in v for c in cls._FIELD_BAD_CHARS):
            print(f"[profile] WARNING: `{name}` contains code characters "
                  f"(first 60 chars: {v[:60]!r}); falling back to default")
            return ""
        # extension check for header/file
        if name == "header" and not v.endswith((".h", ".hpp")):
            print(f"[profile] WARNING: `header={v!r}` lacks .h/.hpp; falling back")
            return ""
        if name == "file" and not v.endswith((".cpp", ".cc", ".cxx")):
            print(f"[profile] WARNING: `file={v!r}` lacks .cpp/.cc/.cxx; falling back")
            return ""
        return v

    @classmethod
    def from_llm(cls, task_name: str, payload: dict[str, Any],
                 backend: str = "base") -> "KernelProfile":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: v for k, v in (payload or {}).items() if k in allowed}
        clean["task_name"] = task_name
        clean["backend"] = backend
        # sanitize identifier-shaped fields BEFORE construction — these are the
        # ones that get plumbed straight into the build system, where a bad
        # value silently corrupts every subsequent round (see Fix A).
        for fld in ("class_name", "header", "file"):
            if fld in clean:
                clean[fld] = cls._sanitize_identifier_field(fld, clean[fld])
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
        # Functional ops carry weights as runtime inputs — they MUST be a
        # multi-input layer (one_blob_only=False) so the LayerOracle / NetOracle
        # / production net all feed weights via std::vector<Mat> bottom_blobs
        # instead of ModelBin. NetOracle compatibility hinges on this: pnnx
        # emits an empty .ncnn.bin for functional ops, so a Cand_X that calls
        # mb.load(...) crashes with "ModelBin read flag_struct failed 0".
        if prof.weights_from_inputs:
            if prof.one_blob_only:
                print(f"[profile] functional op (weights_from_inputs={prof.weights_from_inputs}) "
                      f"→ forcing one_blob_only=False (was True)")
                prof.one_blob_only = False
            if prof.weight_keys:
                print(f"[profile] WARNING: both weights_from_inputs and weight_keys set; "
                      f"clearing weight_keys (functional ops have no state_dict path)")
                prof.weight_keys = []
        return prof

    @property
    def is_multi_input(self) -> bool:
        """True when the op takes multiple bottom_blobs (NOT one_blob_only),
        regardless of whether those extra blobs are weights or activations."""
        return not self.one_blob_only

    @property
    def is_functional(self) -> bool:
        """True for functional ops (F.conv2d / F.linear / ...) where weights
        ride in as forward inputs. Implies is_multi_input."""
        return bool(self.weights_from_inputs)

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
    failure_category: str = ""   # diagnosis-conditioned label (E3/E4/E5/E6...) for stats
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
