"""LLMProposer — turns the baseline kernel into a parameterized template.

Workflow §3 (Proposer = LLM structural layer). The Proposer emits one
ParameterizedTemplate per round: parameterized kernel source + ParamSpec knobs +
LLM-derived physical constraints + rationale. The inner search then measures it.

Parsing is defensive: code fences (filename on the first inner line) → kernel
files; the single ```json block → params/constraints/techniques/rationale. If the
LLM omits the json or a knob, we fall back to an identity template (the baseline
with no knobs) so the loop degrades gracefully instead of crashing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from schemas import ParameterizedTemplate, ParamSpec

_FILE_RE = re.compile(r"[A-Za-z0-9_./+-]+\.(?:cpp|cc|cxx|hpp|h)")
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+]*)\s*\n(.*?)```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def _extract_code(response: str) -> dict[str, str]:
    """{basename: code} for fenced blocks whose first inner line is a filename."""
    code: dict[str, str] = {}
    for m in _FENCE_RE.finditer(response):
        lines = m.group(1).splitlines()
        if not lines:
            continue
        first = lines[0].strip().lstrip("/* ").strip()
        hit = _FILE_RE.fullmatch(first) or _FILE_RE.match(first)
        if hit and "include" not in first:
            from pathlib import Path
            name = Path(hit.group(0)).name
            if name not in code:
                code[name] = "\n".join(lines[1:]).strip() + "\n"
    return code


def _extract_json(response: str) -> dict[str, Any]:
    m = _JSON_FENCE_RE.search(response)
    if not m:
        # last-ditch: any fence that parses as a dict
        for fm in _FENCE_RE.finditer(response):
            try:
                obj = json.loads(fm.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001
                continue
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return {}


def _detect_class_name(code: dict[str, str]) -> str:
    for src in code.values():
        m = re.search(r"class\s+(\w+)\s*:\s*public\s+\w+", src or "")
        if m:
            return m.group(1)
    return ""


def _split_files(code: dict[str, str]) -> tuple[str | None, str | None]:
    cpp = next((n for n in code if n.endswith((".cpp", ".cc", ".cxx"))), None)
    hdr = next((n for n in code if n.endswith((".h", ".hpp"))), None)
    return cpp, hdr


def parse_template(response: str, baseline_kernel: dict[str, str]) -> ParameterizedTemplate:
    """Parse an LLM response into a ParameterizedTemplate.

    Falls back to the baseline (no knobs) when code/json is missing — that makes
    the round a no-op rather than a crash.
    """
    code = _extract_code(response) or dict(baseline_kernel)
    meta = _extract_json(response)

    params: dict[str, ParamSpec] = {}
    for name, spec in (meta.get("params") or {}).items():
        vals = spec.get("values") if isinstance(spec, dict) else spec
        if not vals:
            continue
        params[name] = ParamSpec(
            name=name, values=list(vals),
            dtype=(spec.get("dtype", "int") if isinstance(spec, dict) else "int"),
            desc=(spec.get("desc", "") if isinstance(spec, dict) else ""),
        )

    cpp, hdr = _split_files(code)
    base_cpp, base_hdr = _split_files(baseline_kernel)
    class_name = _detect_class_name(code) or _detect_class_name(baseline_kernel)
    # bd_labels: the LLM's explicit Σ-axis declaration (Method M2.4). Normalize
    # keys/values to lowercase strings; tolerate the LLM omitting it.
    raw_labels = meta.get("bd_labels") or {}
    bd_labels: dict[str, str] = {}
    if isinstance(raw_labels, dict):
        for k, v in raw_labels.items():
            if isinstance(v, str) and v.strip():
                bd_labels[str(k).strip().lower()] = v.strip().lower()
    return ParameterizedTemplate(
        kernel_files=code,
        params=params,
        class_name=class_name,
        header=hdr or base_hdr or "",
        file=cpp or base_cpp or "",
        rationale=str(meta.get("rationale", "")),
        techniques=list(meta.get("techniques", []) or []),
        constraints=list(meta.get("constraints", []) or []),
        bd_labels=bd_labels,
    )


class LLMProposer:
    """Drives an LLM to propose parameterized templates, round by round."""

    def __init__(
        self,
        *,
        task_name: str,
        baseline_kernel: dict[str, str],
        hardware: dict[str, Any],
        llm_query: Callable[[str, str], str],
        model: str = "deepseek-v4-pro",
        backend: str = "base",
        wiki: Any | None = None,        # WikiLoader; None => no wiki context
        regime: str = "unknown",        # roofline regime; drives BD-axis injection
    ) -> None:
        self.task_name = task_name
        self.baseline_kernel = dict(baseline_kernel)
        self.hardware = hardware
        self.llm = llm_query
        self.model = model
        self.backend = backend
        self.wiki = wiki
        self.regime = regime

    def _wiki_context(self) -> str:
        """Retrieve generic primitives + bd_axes[regime] + backend knowledge;
        empty string when disabled."""
        if self.wiki is None:
            return ""
        try:
            return self.wiki.context_block(self.regime) or ""
        except Exception:  # noqa: BLE001
            # A malformed wiki file must NOT crash the optimize loop.
            return ""

    def _sigma_block(self) -> str:
        """The machine-readable Σ axis menu for (backend, regime), rendered for
        the prompt so the LLM declares bd_labels from the real vocabulary
        (Method M2.4). Empty when no wiki_root — the LLM then falls back to
        free-form techniques (keyword classification)."""
        if self.wiki is None or not getattr(self.wiki, "wiki_root", None):
            return ""
        try:
            from policy import sigma as _sigma
            # Σ only carries memory_bound / compute_bound; mixed/unknown → the
            # conservative memory_bound menu (matches wiki's regime aliasing).
            reg = (self.regime or "").strip().lower().replace("-", "_")
            if reg not in ("memory_bound", "compute_bound"):
                reg = "memory_bound"
            sg = _sigma.load(self.wiki.wiki_root, self.backend)
            return sg.render_for_prompt(reg)
        except Exception:  # noqa: BLE001
            return ""

    def propose(self, history: list) -> ParameterizedTemplate:
        from .prompts import proposer_prompt
        tried: list[str] = []
        for it in history:
            tried.extend(getattr(it, "techniques", []) or [])
        prompt = proposer_prompt(
            self.task_name, self.baseline_kernel, self.hardware,
            sorted(set(tried)),
            context=self._wiki_context(), backend=self.backend,
            sigma_block=self._sigma_block(),
        )
        response = self.llm(prompt, self.model)
        return parse_template(response, self.baseline_kernel)

    def vary(self, parent, directive: str, history: list) -> ParameterizedTemplate:
        """MAP-Elites variation (Workflow §7.2 step②). `parent` is an Elite
        (has .kernel_code) or a ParameterizedTemplate (has .kernel_files)."""
        from .prompts import vary_prompt
        parent_code = (getattr(parent, "kernel_code", None)
                       or getattr(parent, "kernel_files", None) or self.baseline_kernel)
        tried: list[str] = []
        fails: list[str] = []
        for it in history:
            if isinstance(it, dict):
                tried.append(it.get("directive", ""))
                fs = it.get("failure_summary") or it.get("error")
                if fs:
                    fails.append(f"[{it.get('directive', '?')}] {fs}")
        prompt = vary_prompt(
            self.task_name, parent_code, self.hardware,
            directive, sorted(set(t for t in tried if t)),
            recent_failures=fails[-3:],
            context=self._wiki_context(), backend=self.backend,
            sigma_block=self._sigma_block(),
        )
        response = self.llm(prompt, self.model)
        return parse_template(response, parent_code)

    def crossover(self, a, b, history: list) -> ParameterizedTemplate:
        """MAP-Elites CROSSOVER: recombine two elites (from different niches) into
        one child. `a`/`b` are Elites (.kernel_code) or templates (.kernel_files)."""
        from .prompts import crossover_prompt

        def _code(x):
            return (getattr(x, "kernel_code", None) or getattr(x, "kernel_files", None)
                    or self.baseline_kernel)

        def _cell(x):
            c = getattr(x, "cell", None)
            return "/".join(map(str, c)) if isinstance(c, (list, tuple)) else ""

        fails = []
        for it in history:
            if isinstance(it, dict):
                fs = it.get("failure_summary") or it.get("error")
                if fs:
                    fails.append(f"[{it.get('directive', '?')}] {fs}")
        prompt = crossover_prompt(
            self.task_name, _code(a), _code(b), self.hardware,
            cell_a=_cell(a), cell_b=_cell(b), recent_failures=fails[-3:],
            context=self._wiki_context(), backend=self.backend,
            sigma_block=self._sigma_block(),
        )
        response = self.llm(prompt, self.model)
        return parse_template(response, _code(a))
