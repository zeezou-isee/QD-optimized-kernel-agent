"""Functional steps for the KernelAgent.

    introspect_model      read input shapes + state_dict (grounding for the LLM)
    extract_kernel_code   parse LLM response -> {filename: code} (.h/.cpp)
    retrieve_layer_example pull a nearest existing ncnn base layer to imitate
    verify_kernel         write files -> LayerOracle.verify(vs PyTorch) -> classify

Verification backend is layer_oracle.LayerOracle (方案A: compile candidate .cpp +
libncnn.a, opt all-off = base kernel, allclose vs PyTorch). Compile errors are
localized via graph_pipeline.locate_build_errors (tree-sitter).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_oracle import LayerOracle, torch_to_ncnn_input
from graph_pipeline import locate_build_errors
from kernel_schemas import KernelProfile, KernelResult


# ---------------------------------------------------------------------------
_FILE_RE = re.compile(r"[A-Za-z0-9_./+-]+\.(?:cpp|cc|cxx|hpp|h)")
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+]*)\s*\n(.*?)```", re.DOTALL)

_ARM_DEGRADE_MSG = (
    "ARM ISOLATION: the arm subclass {cls} does not define its own "
    "forward/forward_inplace, so C++ virtual dispatch silently falls back to the "
    "inherited base CPU kernel — the 'arm' result is a degraded base result (it can "
    "still pass allclose for elementwise ops). Define {cls}::forward (or "
    "{cls}::forward_inplace) with the NEON/NC4HW4 implementation."
)


def arm_forward_overridden(code_book: dict[str, str], arm_class: str) -> bool:
    """True if the arm subclass defines its OWN forward/forward_inplace.

    The arm oracle instantiates the arm class directly (isolated instantiation), but
    the arm forward shares the base's signature `forward(const Mat&, ...)`. If the
    subclass forgets to override it, virtual dispatch uses the inherited *base* CPU
    forward and numeric still passes — i.e. a degraded base kernel masquerading as
    arm. We require an explicit out-of-line override (`<arm_class>::forward[...]`),
    which is the ncnn convention, so the run actually exercises the NEON path.
    """
    pat = re.compile(re.escape(arm_class) + r"\s*::\s*forward")
    return any(pat.search(c or "") for c in code_book.values())


def extract_kernel_code(response: str) -> dict[str, str]:
    """{basename: code} for fenced blocks whose first inner line is a filename."""
    code: dict[str, str] = {}
    # path line right before a fence
    for m in re.finditer(r"(?P<name>" + _FILE_RE.pattern + r")\s*\n```(?:[a-zA-Z0-9_+]*)\s*\n(?P<body>.*?)```", response, re.DOTALL):
        code[Path(m.group("name").strip()).name] = m.group("body").strip() + "\n"
    # filename as first line inside the fence
    for m in _FENCE_RE.finditer(response):
        lines = m.group(1).splitlines()
        if not lines:
            continue
        first = lines[0].strip().lstrip("/* ").strip()
        hit = _FILE_RE.fullmatch(first) or _FILE_RE.match(first)
        if hit and "include" not in first:
            name = Path(hit.group(0)).name
            if name not in code:
                code[name] = "\n".join(lines[1:]).strip() + "\n"
    return code


# ---------------------------------------------------------------------------
def _load_module(model_py: str | Path):
    spec = importlib.util.spec_from_file_location("ds_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_model(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = mod.Model(*init) if init else mod.Model()
    model.eval()
    return model, init


def introspect_model(model_py: str | Path) -> dict[str, Any]:
    """Input shapes + state_dict (key->shape) + init inputs, for grounding."""
    mod = _load_module(model_py)
    model, init = _build_model(mod)
    inputs = mod.get_inputs()
    sd = {k: list(v.shape) for k, v in model.state_dict().items()}
    return {
        "input_shapes": [list(t.shape) for t in inputs],
        "state_dict": sd,
        "init_inputs": _jsonable(init),
    }


def _jsonable(x):
    try:
        import torch
        if isinstance(x, (list, tuple)):
            return [_jsonable(i) for i in x]
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        return x
    except Exception:
        return str(x)


def retrieve_layer_example(ncnn_root: Path, analog: str, max_files: int = 1,
                           backend: str = "base") -> dict[str, str]:
    """Read an ncnn layer as a coding template.

    base   -> src/layer/<analog>.{h,cpp}
    arm    -> src/layer/arm/<analog>_arm.{h,cpp}  PLUS the base src/layer/<analog>.{h,cpp}
              (the arm layer subclasses the base, so both are useful context)
    """
    layer_dir = Path(ncnn_root) / "src" / "layer"
    stem = (analog or "absval").strip().lower().replace("::", "").replace(" ", "")
    # an arm analog may already carry the _arm suffix; normalize to the base stem
    base_stem = stem[:-4] if stem.endswith("_arm") else stem
    out: dict[str, str] = {}

    def _read_into(d: Path, name_stem: str) -> bool:
        hit = False
        for ext in (".h", ".cpp"):
            f = d / f"{name_stem}{ext}"
            if f.exists():
                out[f.name] = f.read_text(encoding="utf-8", errors="replace")
                hit = True
        return hit

    if backend == "arm":
        if not _read_into(layer_dir / "arm", f"{base_stem}_arm"):
            _read_into(layer_dir / "arm", "absval_arm")   # fallback example
        _read_into(layer_dir, base_stem)                  # base for parent-class context
    else:
        if not _read_into(layer_dir, base_stem):
            _read_into(layer_dir, "absval")
    return out


# ---------------------------------------------------------------------------
def verify_kernel(
    oracle: LayerOracle,
    profile: KernelProfile,
    code_book: dict[str, str],
    model_py: str | Path,
    round_dir: Path,
    run_numeric: bool = True,
    tol: float = 2e-3,
    base_files: dict[str, str] | None = None,
    extra_includes: tuple = (),
    packing: int = 0,
) -> KernelResult:
    """Verify a candidate kernel against PyTorch via LayerOracle.

    For an arm backend kernel, pass the verified base layer files via `base_files`
    (written alongside so the arm header's `#include "<base>.h"` resolves and the
    base .cpp is compiled in as an extra source), `extra_includes` pointing at
    `src/layer/arm`, and `packing=4` to exercise the NC4HW4 NEON path.
    """
    res = KernelResult(task_name=profile.task_name, profile=profile.to_dict(),
                       response_code=code_book, identify_ok=True)
    if not code_book:
        res.messages.append("no code extracted")
        return res
    res.generate_ok = True

    # arm isolation precondition: the subclass must override forward, else it
    # silently degrades to the inherited base CPU kernel (enforced at each return).
    arm_degraded = (profile.backend == "arm"
                    and not arm_forward_overridden(code_book, profile.class_name))

    # write candidate files into the round dir
    round_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = None
    for name, content in code_book.items():
        p = round_dir / name
        p.write_text(content, encoding="utf-8")
        if name.endswith((".cpp", ".cc", ".cxx")):
            cpp_path = p
    if cpp_path is None:
        res.compile_error = "no .cpp file among generated files"
        res.messages.append("missing .cpp")
        return res
    res.artifacts["cpp"] = str(cpp_path)

    # arm/vulkan: drop the verified base files next to the candidate (parent class)
    # and compile the base .cpp in as an extra source.
    extra_sources: list[str] = []
    for name, content in (base_files or {}).items():
        p = round_dir / name
        p.write_text(content, encoding="utf-8")
        if name.endswith((".cpp", ".cc", ".cxx")):
            extra_sources.append(str(p))

    # vulkan: the candidate also emits a separate .comp shader (compiled at runtime
    # by the VulkanLayerOracle). Locate it among the written files.
    shader_path = None
    if profile.backend == "vulkan":
        if profile.shader and (round_dir / profile.shader).exists():
            shader_path = round_dir / profile.shader
        else:
            shader_path = next((round_dir / n for n in code_book if n.endswith(".comp")), None)
        if shader_path is None:
            res.compile_error = f"vulkan kernel missing its .comp shader ({profile.shader})"
            res.messages.append("missing .comp")
            return res

    # backend-specific oracle kwargs: vulkan passes the shader; base/arm pass packing
    backend_kwargs: dict = ({"shader": str(shader_path)} if profile.backend == "vulkan"
                            else {"packing": packing})

    # PyTorch reference (single sample, ncnn layout)
    import torch
    mod = _load_module(model_py)
    model, _ = _build_model(mod)
    inputs = mod.get_inputs()
    with torch.no_grad():
        ref = model(*inputs)
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    ref_np = ref.detach().numpy()
    reference = ref_np[0] if ref_np.ndim >= 2 else ref_np
    ncnn_inputs = [torch_to_ncnn_input(t.detach().numpy()) for t in inputs]

    # weights from state_dict in profile order
    sd = model.state_dict()
    weights = []
    for k in profile.weight_keys:
        if k not in sd:
            res.compile_ok = False
            res.compile_error = f"weight key '{k}' not in state_dict {list(sd)}"
            res.messages.append("bad weight_keys")
            return res
        weights.append(sd[k].detach().numpy().reshape(-1))

    params = {int(k): v for k, v in (profile.params or {}).items()}

    if not run_numeric:
        # compile-only check via oracle.run (still needs an input)
        out = oracle.run(candidate_cpp=cpp_path, class_name=profile.class_name,
                         header=profile.header, params=params, inputs=ncnn_inputs, weights=weights,
                         extra_sources=extra_sources, extra_includes=extra_includes, **backend_kwargs)
        res.compile_ok = "compile failed" not in (out.error or "")
        if not res.compile_ok:
            res.compile_error = locate_build_errors(out.compile_log, profile.file.split(".")[0])
        elif arm_degraded:
            res.numeric_ok = False
            res.numeric_log = _ARM_DEGRADE_MSG.format(cls=profile.class_name)
            res.messages.append("arm not overridden (degrades to base)")
        else:
            res.numeric_skipped = True  # compile-only mode
            res.messages.append("numeric skipped")
        return res

    verdict = oracle.verify(candidate_cpp=cpp_path, class_name=profile.class_name,
                            header=profile.header, params=params, inputs=ncnn_inputs,
                            weights=weights, reference=reference, tol=tol,
                            extra_sources=extra_sources, extra_includes=extra_includes, **backend_kwargs)

    # classify
    if verdict.error and "compile failed" in verdict.error:
        res.compile_ok = False
        res.compile_error = locate_build_errors(verdict.compile_log, profile.file.split(".")[0])
        res.messages.append("compile failed")
        return res
    res.compile_ok = True
    # vulkan on a host without a GPU: compiled+linked OK but cannot run -> skip
    # (not a failure). The kernel is accepted as compile-verified.
    if getattr(verdict, "skipped", False):
        res.numeric_skipped = True
        res.numeric_log = verdict.detail or "vulkan device unavailable (skipped)"
        res.messages.append("numeric skipped (no vulkan device)")
        return res
    if not verdict.ok:  # compiled but runner crashed at runtime
        res.numeric_ok = False
        res.numeric_log = "kernel crashed at runtime:\n" + "\n".join(verdict.run_log.splitlines()[-12:])
        res.messages.append("runtime crash")
        return res
    res.max_diff = verdict.max_diff
    res.numeric_ok = bool(verdict.passed)
    res.numeric_log = verdict.detail
    res.messages.append("numeric passed" if verdict.passed else "numeric failed")
    # arm isolation guard: a numeric pass on a non-overriding arm subclass is a
    # false pass (it ran the base kernel). Flip to failure so the loop repairs it.
    if res.numeric_ok and arm_degraded:
        res.numeric_ok = False
        res.numeric_log = _ARM_DEGRADE_MSG.format(cls=profile.class_name)
        res.messages.append("arm not overridden (degrades to base)")
    return res
