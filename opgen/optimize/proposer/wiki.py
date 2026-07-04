"""WikiLoader — retrieves generic optimization knowledge for OptimizeAgent.

v1 architecture (operator-agnostic; drops per-family playbook):

    experience_pool/wiki/
        primitives/                       ← 4 optimization primitive families
        bd_axes/{memory_bound,compute_bound,mixed}.md
                                          ← BD coordinate systems per regime
        heuristics/                       ← cross-framework "where to look first"
        bottleneck/roofline_regimes.md    ← regime classification + early stop
        {arm,vulkan}/
            hardware/*.json               ← hw_ns extras (extended symbols)
            backend/dialect.md            ← how to speak the backend
            backend/idioms.md             ← framework conventions (ncnn)
            backend/failure_patterns.md   ← oracle E-code preemption

Retrieval key changed from (backend, family) to (backend, regime). The LLM
receives methodology (primitives + BD axes + backend language), not per-op
recipes. The proposer chooses where to explore in the space rather than
picking from a menu we handed it.

Contract:
    WikiLoader(wiki_root, backend)              # backend ∈ {arm, vulkan}
        .context_block(regime: str) -> str      # 4-section markdown for the prompt
        .hardware_extras(hw_key: str) -> dict   # extra symbols to merge into hw_ns
        ._check_symbols() -> list[str]          # dev-time guardrail

The loader is deliberately dumb: it reads files, checks caps, concatenates.
All structure is in the filesystem. If a wiki file is missing that section
is quietly skipped (empty) — a missing file must degrade gracefully, not
crash the loop.

Prompt-length policy (enforced by the loader):
    each primitive page          <= 250 lines
    bd_axes page (per regime)    <= 250 lines
    roofline_regimes             <= 150 lines
    dialect page                 <= 400 lines
    idioms page                  <= 200 lines
    failure_patterns             <=  80 lines
    total context_block          <= 3000 lines  (truncated with marker)
"""

from __future__ import annotations

import re
from pathlib import Path

_BACKENDS = ("arm", "vulkan")

_PRIMITIVE_FILES = (
    "reduce_compute.md",
    "reduce_memory_traffic.md",
    "increase_parallelism.md",
    "hardware_specialized.md",
)

_HEURISTIC_FILES = (
    "tiling_and_packing.md",
    "algorithm_selection.md",
    "precision_and_quant.md",
    "parallelism_and_workgroup.md",
)

_REGIME_ALIASES = {
    "memory_bound": "memory_bound",
    "memorybound": "memory_bound",
    "memory-bound": "memory_bound",
    "compute_bound": "compute_bound",
    "computebound": "compute_bound",
    "compute-bound": "compute_bound",
    "mixed": "mixed",
    "unknown": "mixed",     # unknown regime → conservative mixed treatment
    "": "mixed",
}

# hw_ns symbols the wiki is allowed to mention in constraints. Kept in lockstep
# with ConstraintEngine.hw_ns + the extras merged by WikiLoader.hardware_extras.
_ALLOWED_SYMBOLS_ARM = {
    "L1", "L1D", "L2", "VEC_BITS", "FP32_PER_VEC", "VECTOR_REGS", "NEON",
    "L3", "CACHE_LINE",
    "FMLA_LATENCY", "FMLA_THROUGHPUT",
    "HAS_DOTPROD", "HAS_ASIMDHP", "HAS_BF16", "HAS_I8MM",
}
_ALLOWED_SYMBOLS_VULKAN = {
    "L1", "L1D", "L2", "VEC_BITS", "FP32_PER_VEC", "VECTOR_REGS", "NEON",
    "SUBGROUP_SIZE", "MAX_SHARED_MEM_BYTES", "MAX_WG_INVOCATIONS",
    "MAX_PUSH_CONSTANTS_BYTES",
    "HAS_FP16", "HAS_INT8", "HAS_COOPMAT", "HAS_SUBGROUP_ARITHMETIC",
    "HAS_SUBGROUP_SHUFFLE", "HAS_SUBGROUP_BALLOT",
}

_MAX_PRIMITIVE_LINES = 250
_MAX_BD_LINES = 250
_MAX_HEURISTIC_LINES = 200
_MAX_ROOFLINE_LINES = 150
_MAX_DIALECT_LINES = 400
_MAX_IDIOMS_LINES = 200
_MAX_FAILURE_LINES = 80
_MAX_TOTAL_LINES = 3600


def _clip_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n<!-- truncated -->\n"


def _read_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _normalize_regime(regime: str) -> str:
    return _REGIME_ALIASES.get((regime or "").strip().lower(), "mixed")


class WikiLoader:
    def __init__(self, wiki_root: Path | str, backend: str) -> None:
        backend = (backend or "").lower()
        self.wiki_root = Path(wiki_root)
        if backend not in _BACKENDS:
            self.backend = backend
            self.enabled = False
            return
        self.backend = backend
        self.enabled = self.wiki_root.exists()

    # ------------------------------------------------------------------ prompt
    def context_block(self, regime: str = "unknown") -> str:
        """Return a 4-section markdown block: primitives + bd_axes[regime] +
        roofline_regimes + backend/{dialect,idioms,failure_patterns}.

        Empty string when the loader is disabled OR when the primitives dir
        is missing (a clean fallback — caller then omits the section rather
        than crashing).
        """
        if not self.enabled:
            return ""
        parts: list[str] = []

        # 1. Optimization primitives (all 4, always injected)
        prim_dir = self.wiki_root / "primitives"
        prim_bodies: list[str] = []
        for fname in _PRIMITIVE_FILES:
            body = _read_or_empty(prim_dir / fname)
            if body.strip():
                prim_bodies.append(_clip_lines(body, _MAX_PRIMITIVE_LINES))
        if prim_bodies:
            parts.append("# Optimization primitives\n" + "\n\n---\n\n".join(prim_bodies))

        # 2. BD coordinate system for the resolved regime
        norm_regime = _normalize_regime(regime)
        bd_body = _read_or_empty(self.wiki_root / "bd_axes" / f"{norm_regime}.md")
        if bd_body.strip():
            parts.append(f"# Search space (BD axes) — {norm_regime}\n"
                         + _clip_lines(bd_body, _MAX_BD_LINES))

        # 3. Cross-framework heuristics — "where to look first" (all 4, always)
        heur_dir = self.wiki_root / "heuristics"
        heur_bodies: list[str] = []
        for fname in _HEURISTIC_FILES:
            body = _read_or_empty(heur_dir / fname)
            if body.strip():
                heur_bodies.append(_clip_lines(body, _MAX_HEURISTIC_LINES))
        if heur_bodies:
            parts.append("# Search heuristics (cross-framework priors)\n"
                         + "\n\n---\n\n".join(heur_bodies))

        # 4. Roofline / regime classification (always injected)
        roofline_body = _read_or_empty(self.wiki_root / "bottleneck" / "roofline_regimes.md")
        if roofline_body.strip():
            parts.append("# Roofline & regime rules\n"
                         + _clip_lines(roofline_body, _MAX_ROOFLINE_LINES))

        # 5. Backend dialect + idioms + failure patterns
        backend_dir = self.wiki_root / self.backend / "backend"
        dialect = _read_or_empty(backend_dir / "dialect.md")
        if dialect.strip():
            parts.append(f"# Backend dialect (how to speak {self.backend})\n"
                         + _clip_lines(dialect, _MAX_DIALECT_LINES))
        idioms = _read_or_empty(backend_dir / "idioms.md")
        if idioms.strip():
            parts.append(f"# Framework idioms ({self.backend})\n"
                         + _clip_lines(idioms, _MAX_IDIOMS_LINES))
        failures = _read_or_empty(backend_dir / "failure_patterns.md")
        if failures.strip():
            parts.append("# Failure codes to preempt\n"
                         + _clip_lines(failures, _MAX_FAILURE_LINES))

        if not parts:
            return ""
        return _clip_lines("\n\n".join(parts), _MAX_TOTAL_LINES)

    # -------------------------------------------------------------- hw extras
    def hardware_extras(self, hw_key: str = "apple_m5") -> dict[str, float | int]:
        """Load extra `hw_ns` symbols for the given hardware profile.

        Returns {} when the loader is disabled or the profile is missing —
        ConstraintEngine falls back to its 4-key default namespace and nothing
        breaks; the LLM just doesn't get to reference the extra symbols.
        """
        if not self.enabled:
            return {}
        prof_path = self.wiki_root / self.backend / "hardware" / f"{hw_key}.json"
        if not prof_path.exists():
            return {}
        import json
        try:
            data = json.loads(prof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if self.backend == "arm":
            return self._arm_extras(data)
        if self.backend == "vulkan":
            return self._vulkan_extras(data)
        return {}

    def _arm_extras(self, data: dict) -> dict[str, float | int]:
        cache = data.get("cache", {})
        features = data.get("features", {})
        pipeline = data.get("pipeline", {})
        out: dict[str, float | int] = {}
        if "l3_bytes" in cache:
            out["L3"] = int(cache["l3_bytes"])
        if "cache_line_bytes" in cache:
            out["CACHE_LINE"] = int(cache["cache_line_bytes"])
        for k, sym in (
            ("has_dotprod", "HAS_DOTPROD"),
            ("has_fp16", "HAS_ASIMDHP"),
            ("has_bf16", "HAS_BF16"),
            ("has_i8mm", "HAS_I8MM"),
        ):
            if k in features:
                out[sym] = 1 if features[k] else 0
        if "fmla_latency" in pipeline:
            out["FMLA_LATENCY"] = float(pipeline["fmla_latency"])
        if "fmla_throughput" in pipeline:
            out["FMLA_THROUGHPUT"] = float(pipeline["fmla_throughput"])
        return out

    def _vulkan_extras(self, data: dict) -> dict[str, float | int]:
        limits = data.get("compute_limits", {})
        sg = data.get("subgroup", {})
        features = data.get("features", {})
        exts = data.get("extensions_present", {})
        out: dict[str, float | int] = {}
        if "subgroup_size" in sg:
            out["SUBGROUP_SIZE"] = int(sg["subgroup_size"])
        if "max_compute_shared_memory_size" in limits:
            out["MAX_SHARED_MEM_BYTES"] = int(limits["max_compute_shared_memory_size"])
        if "max_compute_workgroup_invocations" in limits:
            out["MAX_WG_INVOCATIONS"] = int(limits["max_compute_workgroup_invocations"])
        if "max_push_constants_size" in limits:
            out["MAX_PUSH_CONSTANTS_BYTES"] = int(limits["max_push_constants_size"])
        if "shader_float16" in features:
            out["HAS_FP16"] = 1 if features["shader_float16"] else 0
        if "shader_int8" in features:
            out["HAS_INT8"] = 1 if features["shader_int8"] else 0
        if "VK_KHR_cooperative_matrix" in exts:
            out["HAS_COOPMAT"] = 1 if exts["VK_KHR_cooperative_matrix"] else 0
        ops = set(sg.get("subgroup_supported_operations", []))
        if ops:
            out["HAS_SUBGROUP_ARITHMETIC"] = 1 if "VK_SUBGROUP_FEATURE_ARITHMETIC_BIT" in ops else 0
            out["HAS_SUBGROUP_SHUFFLE"] = 1 if "VK_SUBGROUP_FEATURE_SHUFFLE_BIT" in ops else 0
            out["HAS_SUBGROUP_BALLOT"] = 1 if "VK_SUBGROUP_FEATURE_BALLOT_BIT" in ops else 0
        return out

    # ------------------------------------------------------------ dev check
    def _check_symbols(self) -> list[str]:
        """Return the list of ALL-CAPS tokens appearing in the dialect page that
        aren't in the allowed symbol set for this backend. Zero-length list ==
        wiki is in sync with ConstraintEngine's namespace.

        Used at test time — not called in the hot path. LLM may still reference
        these tokens; ConstraintEngine silently skips constraints it can't eval,
        so this is a "you probably meant something else" warning, not a fatal.
        """
        if not self.enabled:
            return []
        allowed = _ALLOWED_SYMBOLS_ARM if self.backend == "arm" else _ALLOWED_SYMBOLS_VULKAN
        dialect_path = self.wiki_root / self.backend / "backend" / "dialect.md"
        text = _read_or_empty(dialect_path)
        if not text:
            return []
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`\n]+`", "", text)
        tokens = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
        return sorted(tokens - allowed)
