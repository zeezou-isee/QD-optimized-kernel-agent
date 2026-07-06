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

from layer_oracle import LayerOracle, pnnx_driven_ncnn_inputs, torch_to_ncnn_input
from graph_pipeline import locate_build_errors
from config import RUNS_ROOT
from kernel_schemas import KernelProfile, KernelResult


# ---------------------------------------------------------------------------
_FILE_RE = re.compile(r"[A-Za-z0-9_./+-]+\.(?:cpp|cc|cxx|hpp|h|comp)")
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
    import torch
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    # Fixed seed so the weights here match the exported .ncnn.bin and every other
    # numeric-reference model in the pipeline (see graph_pipeline make_pt).
    torch.manual_seed(0)
    model = mod.Model(*init) if init else mod.Model()
    model.eval()
    return model, init


def introspect_model(model_py: str | Path) -> dict[str, Any]:
    """Input shapes + state_dict (key->shape) + init inputs + EXPECTED OUTPUT shape.

    The output shape is the explicit shape contract (8.3): the kernel's forward must
    produce exactly this (batch dropped) ncnn Mat shape.
    """
    import torch
    mod = _load_module(model_py)
    model, init = _build_model(mod)
    inputs = mod.get_inputs()
    sd = {k: list(v.shape) for k, v in model.state_dict().items()}
    out_shape = None
    ncnn_out_shape = None
    try:
        with torch.no_grad():
            out = model(*inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out_shape = list(out.shape)
        ncnn_out_shape = out_shape[1:] if len(out_shape) >= 2 else out_shape
    except Exception:  # noqa: BLE001
        pass
    return {
        "input_shapes": [list(t.shape) for t in inputs],
        "state_dict": sd,
        "init_inputs": _jsonable(init),
        "output_shape": out_shape,
        "ncnn_output_shape": ncnn_out_shape,
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
    vulkan -> src/layer/vulkan/<analog>_vulkan.{h,cpp}
              PLUS src/layer/vulkan/shader/<analog>.comp (the actual shader math)
              PLUS the base src/layer/<analog>.{h,cpp}
              (host code + shader dialect + parent-class semantics — all three)
              If a pack4 variant exists we ALSO include <analog>_pack4.comp so the
              LLM has seen the sfpvec4/buffer_ld4 pattern even though v1 authors
              elempack=1 shaders (helps with SPIR-V mental model + future pack
              extensions). Falls back to absval on any lookup miss so the prompt
              never contains "(no example retrieved)" for vulkan runs.
    """
    layer_dir = Path(ncnn_root) / "src" / "layer"
    stem = (analog or "absval").strip().lower().replace("::", "").replace(" ", "")
    # an arm/vulkan analog may already carry the backend suffix; normalize
    for suf in ("_arm", "_vulkan"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    base_stem = stem
    out: dict[str, str] = {}

    def _read_into(d: Path, name_stem: str, exts: tuple[str, ...] = (".h", ".cpp")) -> bool:
        hit = False
        for ext in exts:
            f = d / f"{name_stem}{ext}"
            if f.exists():
                out[f.name] = f.read_text(encoding="utf-8", errors="replace")
                hit = True
        return hit

    if backend == "arm":
        if not _read_into(layer_dir / "arm", f"{base_stem}_arm"):
            _read_into(layer_dir / "arm", "absval_arm")   # fallback example
        _read_into(layer_dir, base_stem)                  # base for parent-class context
    elif backend == "vulkan":
        vk_dir = layer_dir / "vulkan"
        shader_dir = vk_dir / "shader"
        # 1) host-side <analog>_vulkan.{h,cpp} — pipeline lifecycle + dispatch
        got_host = _read_into(vk_dir, f"{base_stem}_vulkan")
        if not got_host:
            _read_into(vk_dir, "absval_vulkan")           # fallback host example
        # 2) shader body — the actual compute in ncnn shader dialect
        got_shader = _read_into(shader_dir, base_stem, exts=(".comp",))
        if not got_shader:
            _read_into(shader_dir, "absval", exts=(".comp",))
        # 3) pack4 shader (extra vocabulary — sfpvec4 / buffer_ld4)
        _read_into(shader_dir, f"{base_stem}_pack4", exts=(".comp",))
        # 4) base CPU layer — reference semantics the shader must match
        _read_into(layer_dir, base_stem)
    else:
        if not _read_into(layer_dir, base_stem):
            _read_into(layer_dir, "absval")
    return out


# ---------------------------------------------------------------------------
def _size_variants(inputs):
    """Same-rank, smaller-size variants of the model inputs, made by SLICING (keeps
    values in the op's valid domain — random tensors would break log/sqrt/det etc.).

    Yields up to two variants:
      1. HALVED — last two non-batch axes halved (general shape scrutiny)
      2. CHANNEL-GAP — last spatial axis trimmed so `prod(spatial) % 4 != 0`
         (forces ncnn cstep > w*h*d → channel gap, exposes flat-cast bugs
          where the kernel casts Mat to a flat float* across channels)

    Returns [] if no variant could be constructed.
    """
    out = []
    # --- 1) halved variant ---
    halved = []
    changed = False
    for t in inputs:
        s = list(t.shape)
        idx = [slice(None)] * len(s)
        for ax in (len(s) - 1, len(s) - 2):
            if ax >= 1 and s[ax] >= 2:        # keep axis 0 (batch) fixed
                idx[ax] = slice(0, s[ax] // 2)
                changed = True
        halved.append(t[tuple(idx)].contiguous())
    if changed:
        out.append(tuple(halved))

    # --- 2) channel-gap variant ---
    # Pick a slice where w*h*d % 4 != 0 (when packing=elempack=1 the per-channel
    # stride is rounded up to a multiple of 4 floats, so a non-mod-4 spatial product
    # creates a gap between channels that any (float*)mat flat-cast will trip over).
    # ncnn Mat layout: dims=3→w+h+c, dims=4→w+h+d+c.
    gap = []
    forced = False
    odd_choices = (3, 5, 7, 9, 11, 13)
    for t in inputs:
        s = list(t.shape)
        if len(s) < 3:
            # 1D/2D ncnn Mat has no channel dim → no cstep gap mechanism
            gap.append(t)
            continue
        spatial_axes = list(range(2, len(s)))
        # try the smallest pair/triple of odd lengths whose product is not div by 4
        chosen = None
        if len(spatial_axes) == 1:
            for w in odd_choices:
                if w < s[spatial_axes[0]] and w % 4 != 0:
                    chosen = [w]; break
        elif len(spatial_axes) == 2:
            for a in odd_choices:
                if a >= s[spatial_axes[0]]:
                    continue
                for b in odd_choices:
                    if b >= s[spatial_axes[1]]:
                        continue
                    if (a * b) % 4 != 0:
                        chosen = [a, b]; break
                if chosen:
                    break
        else:  # 3 spatial axes (5D torch → 4D ncnn) — trim all three
            for a in odd_choices:
                if a >= s[spatial_axes[0]]:
                    continue
                for b in odd_choices:
                    if b >= s[spatial_axes[1]]:
                        continue
                    for c in odd_choices:
                        if c >= s[spatial_axes[2]]:
                            continue
                        if (a * b * c) % 4 != 0:
                            chosen = [a, b, c]; break
                    if chosen:
                        break
                if chosen:
                    break
        if chosen is not None:
            idx = [slice(None)] * len(s)
            for ax, w in zip(spatial_axes, chosen):
                idx[ax] = slice(0, w)
            gap.append(t[tuple(idx)].contiguous())
            forced = True
        else:
            gap.append(t)
    # skip if the gap variant would be a duplicate of the halved variant
    halved_shapes = tuple(h.shape for h in halved) if changed else None
    if forced and tuple(g.shape for g in gap) != halved_shapes:
        out.append(tuple(gap))
    return out


def _multishape_check(oracle, profile, cpp_path, params, model_py,
                      extra_sources, extra_includes, backend_kwargs, tol,
                      in0_squeezed: bool = False):
    """Re-verify a weightless kernel on size variants (8.2). Returns (category, detail)
    of the first failing variant, or None if all pass / the op rejects them.

    _size_variants yields variants in order: (1) halved, (2) channel-gap. The
    channel-gap variant specifically catches the cstep flat-cast bug — when it's
    what fails, the message points the LLM at the right root cause.

    `in0_squeezed` MUST match the main verify path's squeeze decision: only drop
    axis 0 from inputs/reference when pnnx actually squeezed it (batched nn.Module
    op). For ops where axis 0 is a real matrix dim (2D transpose `ij->ji`,
    multi-input Concat along axis 0), dropping it fabricates a degenerate variant
    and a FALSE failure — the historical bug that failed Einsum_transpose/Concat
    even though their primary shape passed. Variants are same-rank slices, so the
    original op's squeeze decision applies unchanged.
    """
    import torch
    mod = _load_module(model_py)
    model, _ = _build_model(mod)
    variants = _size_variants(mod.get_inputs())
    for vidx, vin in enumerate(variants):
        try:
            with torch.no_grad():
                ref = model(*vin)
            if isinstance(ref, (tuple, list)):
                ref = ref[0]
            ref_np = ref.detach().numpy()
            # mirror the main path: squeeze axis 0 ONLY if pnnx did on the original
            reference = ref_np[0] if (in0_squeezed and ref_np.ndim >= 2) else ref_np
            ncnn_inputs = []
            for t in vin:
                a = np.ascontiguousarray(t.detach().numpy())
                ncnn_inputs.append(a[0] if (in0_squeezed and a.ndim >= 2) else a)
        except Exception:  # noqa: BLE001 — op rejects this shape, skip it
            continue
        v = oracle.verify(candidate_cpp=cpp_path, class_name=profile.class_name,
                          header=profile.header, params=params, inputs=ncnn_inputs,
                          weights=(), reference=reference, tol=tol, backend=profile.backend,
                          extra_sources=extra_sources, extra_includes=extra_includes, **backend_kwargs)
        if getattr(v, "skipped", False):
            return None  # vulkan w/o GPU — can't judge variants
        if not (v.ok and v.passed):
            shp = tuple(int(x) for x in vin[0].shape)
            cat = getattr(v, "failure_category", "") or "E6_VALUE_NUMERICAL"
            # vidx==1 is the channel-gap variant (cstep > w*h*d → flat-cast bug)
            is_gap = (vidx == 1) or (len(shp) >= 2 and (shp[-1] * (shp[-2] if len(shp) >= 3 else 1)) % 4 != 0)
            if is_gap:
                hint = (f"MULTI-SHAPE / CHANNEL-GAP: this variant (input {shp}) has "
                        f"spatial w*h*d NOT divisible by 4, so ncnn pads `mat.cstep` "
                        f"per-channel with unused floats. Your forward most likely casts "
                        f"the Mat to a flat float* and walks w*h*c contiguously — that "
                        f"reads/writes the gap, corrupting channel boundaries. Fix: iterate "
                        f"per-channel via `mat.channel(q)` and process w*h*d elements "
                        f"inside each channel; NEVER `const float* p = (const float*)mat;` "
                        f"across channels. {v.detail}")
            else:
                hint = (f"MULTI-SHAPE: passed the model's shape but FAILED a size variant "
                        f"(input {shp}) — your indexing is hardcoded to one shape. {v.detail}")
            return (cat, hint)
    return None


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
    device_verify: str = "off",       # off | auto | on — run the on-phone gate after host passes
    device_simpleperf: bool = False,  # also collect PMU on device (default off)
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

    # vulkan: the candidate also emits one OR MORE .comp shaders (each compiled
    # at runtime by the VulkanLayerOracle). The PRIMARY shader is the one named
    # by `profile.shader`; any additional .comp files in the code_book are
    # `extra_shaders` (multi-shader ops like BinaryOp / Convolution / etc.).
    shader_path = None
    extra_shaders: list[str] = []
    native_vk = profile.backend == "vulkan" and getattr(profile, "native_vulkan", False)
    if profile.backend == "vulkan" and not native_vk:
        all_comps = [round_dir / n for n in code_book if n.endswith(".comp")]
        if profile.shader and (round_dir / profile.shader).exists():
            shader_path = round_dir / profile.shader
        elif all_comps:
            shader_path = all_comps[0]
        if shader_path is None:
            res.compile_error = f"vulkan kernel missing its .comp shader ({profile.shader})"
            res.messages.append("missing .comp")
            return res
        extra_shaders = [str(p) for p in all_comps if p != shader_path]

    # backend-specific oracle kwargs: vulkan passes the shader (None for a native
    # subclass, which inherits ncnn's baked shader); base/arm pass packing.
    if profile.backend == "vulkan":
        backend_kwargs = {
            "shader": (str(shader_path) if shader_path else None),
            "extra_shaders": extra_shaders,
        }
    else:
        backend_kwargs = {"packing": packing}

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
    # `reference` will be finalised AFTER the input pipeline below decides
    # whether to drop the batch dim — keep the policies symmetric.
    reference = None

    # Functional ops (F.conv2d / F.linear / ...): weights ride in as ADDITIONAL
    # bottom_blobs, NOT via mb.load. This matches what pnnx writes to .ncnn.param
    # (multi-input wiring + empty .ncnn.bin) and avoids the "ModelBin read
    # flag_struct failed 0" crash at NetOracle time. So every input — activation
    # AND weight — goes through ncnn_inputs; oracle.verify gets weights=[].
    wfi = list(getattr(profile, "weights_from_inputs", None) or [])
    if wfi and any(i < 0 or i >= len(inputs) for i in wfi):
        res.compile_ok = False
        res.compile_error = (f"weights_from_inputs={wfi} out of range for "
                             f"{len(inputs)} inputs")
        res.messages.append("bad weights_from_inputs")
        return res
    # Per-input layout: use the SAME pnnx-driven squeeze policy that NetOracle
    # uses, so the LLM sees the same input shapes here that it will see when
    # the kernel is later wired into ncnn::Net. Otherwise LayerOracle silently
    # gives a "drop-batch" 1D input that hides 2D-batch handling bugs (the
    # classic Gemm case: LayerOracle sees (512,), kernel writes a 1D-only
    # forward, NetOracle then feeds (32,512) and the LLM never had a chance
    # to write the 2D branch).
    #
    # Search for the pnnx-emitted _ncnn.py in the OperatorAgent's baseline
    # probe dir (always populated before KernelAgent starts). When absent
    # (kernel-only / standalone), fall back to the old torch_to_ncnn_input
    # heuristic per input.
    # Candidate _ncnn.py locations, in priority order. The OperatorAgent's
    # baseline probe (populated when OperatorAgent has run for this task) is
    # the most authoritative. Failing that, look at the per-backend lazy pnnx
    # probes that KernelAgent may have written during a kernel-only run —
    # kernel_vulkan/_pnnx_probe is what _ensure_baseline_probe drops for the
    # vulkan-native-subclass path (see kernel_agent.py). Base/arm entries are
    # placeholders in case those backends adopt the same lazy probe later; they
    # cost nothing when the dir is missing.
    # NEW-layout canonical location + legacy fallbacks. `analyze/pnnx_probe`
    # is where KernelAgent._ensure_baseline_probe writes today (shared across
    # backends). `operator/_baseline_probe` is OperatorAgent's legacy dir
    # (still active). The `kernel*/_pnnx_probe` entries are pre-migration
    # legacy and drop out naturally once scripts/migrate_runs_layout.py
    # has been run.
    _PROBE_SUBS = ("analyze/pnnx_probe",
                   "operator/_baseline_probe",
                   "kernel_vulkan/_pnnx_probe",
                   "kernel/_pnnx_probe",
                   "kernel_arm/_pnnx_probe",
                   "base_kernel/_pnnx_probe",
                   "backends/vulkan/kernel/_pnnx_probe",
                   "backends/arm/kernel/_pnnx_probe")
    ncnn_py_path = None
    for sub in _PROBE_SUBS:
        for p in (RUNS_ROOT / profile.task_name / sub).rglob("*_ncnn.py"):
            ncnn_py_path = str(p)
            break
        if ncnn_py_path:
            break
    activation_inputs = [t for i, t in enumerate(inputs) if i not in wfi]
    activation_names = [f"in{i}" for i in range(len(activation_inputs))]
    if ncnn_py_path:
        # parse_ncnn_io would give us the real blob names, but the file may
        # use different names. Try the names the .ncnn.py uses verbatim.
        try:
            from layer_oracle.oracle import parse_pnnx_input_squeeze
            policy = parse_pnnx_input_squeeze(ncnn_py_path)
            if policy:
                activation_names = list(policy.keys())[:len(activation_inputs)]
        except Exception:
            pass
        activation_ncnn = pnnx_driven_ncnn_inputs(
            activation_inputs, activation_names, ncnn_py_path)
    else:
        activation_ncnn = [torch_to_ncnn_input(t.detach().numpy())
                           for t in activation_inputs]

    ncnn_inputs = []
    ai = 0
    for i, t in enumerate(inputs):
        if i in wfi:
            ncnn_inputs.append(np.ascontiguousarray(t.detach().numpy()))  # weight raw
        else:
            ncnn_inputs.append(activation_ncnn[ai]); ai += 1

    # Finalise reference shape symmetrically with the input policy. If the FIRST
    # activation got its axis 0 squeezed by pnnx (typical batched nn.Module op),
    # squeeze the reference too. Otherwise (Gemm-like ops where axis 0 is M not
    # a real batch — pnnx writes _ncnn.py with no squeeze), keep the full ref.
    first_act_idx = next((i for i in range(len(inputs)) if i not in wfi), 0)
    first_act_t = inputs[first_act_idx]
    first_act_ncnn = ncnn_inputs[first_act_idx]
    in0_squeezed = (first_act_t.ndim >= 2
                    and first_act_ncnn.ndim == first_act_t.ndim - 1)
    if in0_squeezed and ref_np.ndim >= 2:
        reference = ref_np[0]
    else:
        reference = ref_np

    # Standard nn.Module path: state_dict-backed weights. Mutually exclusive
    # with wfi (KernelProfile.from_llm clears weight_keys when wfi is set).
    weights: list = []
    sd = model.state_dict()
    for k in profile.weight_keys:
        if k not in sd:
            res.compile_ok = False
            res.compile_error = f"weight key '{k}' not in state_dict {list(sd)}"
            res.messages.append("bad weight_keys")
            return res
        weights.append(sd[k].detach().numpy().reshape(-1))

    # Per-weight bin layout flags so the LayerOracle bin is byte-identical to the
    # real .ncnn.bin: 0 = tagged (ncnn fwrite_weight_tag_data, read with type 0),
    # 1 = raw (fwrite_weight_data, read with type 1). Pulled from the ncnn
    # interface dict's weights_load_order, aligned positionally to weight_keys.
    # Defaults to 0 (tagged) for unknown layers / extra slots — preserving prior
    # behavior. This is what makes BatchNorm-style all-secondary-weight layers
    # (slope/mean/var/bias, all type 1) validate correctly instead of misreading
    # a phantom tag as data (var=0 -> 1/sqrt(eps) blowup).
    weight_flags: list[int] = []
    try:
        # `from lookup import ...` used to silently fail (no `lookup` top-level
        # module) — the except swallowed the ImportError and every weight got
        # flag=0. That corrupted every all-secondary-weight layer (BatchNorm's
        # slope/mean/var/bias are all type 1: raw fp32 with NO 4-byte tag). The
        # runner then read `[fp32_of_first_weight_value] ...` as `[tag][raw...]`,
        # dropping the first float and shifting the rest — var≈0 → 1/sqrt(eps)
        # explodes → 16 rounds of nonsense. Import from the actual package path.
        from ncnn_interface.lookup import get_interface
        _iface = get_interface(profile.analog_layer)
        _wlo = (_iface or {}).get("weights_load_order") or []
        for i in range(len(weights)):
            weight_flags.append(int(_wlo[i]["flag"]) if i < len(_wlo) else 0)
    except Exception:  # noqa: BLE001
        weight_flags = [0] * len(weights)

    params = {int(k): v for k, v in (profile.params or {}).items()}

    if not run_numeric:
        # compile-only check via oracle.run (still needs an input)
        out = oracle.run(candidate_cpp=cpp_path, class_name=profile.class_name,
                         header=profile.header, params=params, inputs=ncnn_inputs, weights=weights,
                         weight_flags=weight_flags,
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
                            weights=weights, weight_flags=weight_flags,
                            reference=reference, tol=tol, backend=profile.backend,
                            extra_sources=extra_sources, extra_includes=extra_includes, **backend_kwargs)

    # classify
    # Match both LayerOracle ("compile failed") and VulkanLayerOracle
    # ("vulkan runner build failed" / "vulkan runner cmake configure failed").
    # Missing the vulkan variants used to mark a build failure as compile_ok=True
    # + numeric_log="kernel crashed at runtime:\n" (empty run_log), which hid
    # e.g. the lowercase `binaryop_vulkan` class-name typo for 15 rounds.
    _COMPILE_FAIL_MARKERS = ("compile failed", "vulkan runner build failed",
                             "vulkan runner cmake configure failed")
    if verdict.error and any(m in verdict.error for m in _COMPILE_FAIL_MARKERS):
        res.compile_ok = False
        res.compile_error = locate_build_errors(verdict.compile_log or verdict.error,
                                                profile.file.split(".")[0])
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
        # Include verdict.error verbatim (carries `category=...` marker + the
        # sliced glslang error block on shader_compile paths) so the debugger
        # prompt classifier + shader-focused hints fire. Then attach last-12
        # stderr lines as fallback context for pipeline/dispatch failures.
        err_line = verdict.error or ""
        tail = "\n".join((verdict.run_log or "").splitlines()[-12:])
        res.numeric_log = ("kernel crashed at runtime:\n" + err_line
                           + ("\n\n" + tail if tail else ""))
        res.messages.append("runtime crash")
        return res
    res.max_diff = verdict.max_diff
    res.numeric_ok = bool(verdict.passed)
    res.numeric_log = verdict.detail
    res.failure_category = getattr(verdict, "failure_category", "")
    res.messages.append("numeric passed" if verdict.passed else "numeric failed")
    # arm isolation guard: a numeric pass on a non-overriding arm subclass is a
    # false pass (it ran the base kernel). Flip to failure so the loop repairs it.
    if res.numeric_ok and arm_degraded:
        res.numeric_ok = False
        res.numeric_log = _ARM_DEGRADE_MSG.format(cls=profile.class_name)
        res.messages.append("arm not overridden (degrades to base)")

    # 8.1 differential framing: the base kernel was already verified == PyTorch, so an
    # arm/vulkan numeric failure is a PORT bug (algorithm OK, error in the backend path).
    if (not res.numeric_ok) and (not res.numeric_skipped) and profile.backend in ("arm", "vulkan"):
        res.numeric_log = (f"PORT BUG: the base kernel for this op is already verified == PyTorch, "
                           f"so this {profile.backend} kernel's failure is in the {profile.backend} "
                           f"path (the algorithm is correct — fix the "
                           f"{'NEON/packing' if profile.backend == 'arm' else 'shader/dispatch'} "
                           f"port, not the math).\n" + (res.numeric_log or ""))

    # 8.2 multi-shape scrutiny: a numeric PASS on the single model shape can still hide a
    # shape-specific indexing bug. For weightless ops (input dims not tied to weights),
    # re-verify on a couple of same-rank size variants; any failure flips the verdict.
    # Skip when the op has ANY weights (either nn.Module state_dict keys OR functional
    # weights routed from inputs) — slicing only the activation breaks the math.
    has_weights = bool(profile.weight_keys) or bool(getattr(profile, "weights_from_inputs", None))
    # Native-vulkan subclasses (Cand_X_vulkan : public ncnn::<X>_vulkan {}) hard-code
    # the ncnn params they were loaded with (Reshape's target `w=32768,h=16`,
    # Softmax's axis, ...). A same-rank size-variant that changes the input's
    # element count reasonably fails a Reshape whose target is fixed — that's not
    # a port bug, it's the model contract. Skip multi-shape scrutiny for the
    # native path; the model's own shape has already been verified against PyTorch.
    #
    # FROM-SCRATCH vulkan (agent-written .comp shader): multi-shape scrutiny is
    # EXPLICITLY ENABLED — it is the primary net against the shader-authoring
    # failure modes (hard-coded workgroup dims, forgotten psc(w/h/c), 1D dispatch
    # over a 3D shape, cstep vs w*h*d confusion). We keep the base gates
    # (`not has_weights`, `run_numeric`) but do NOT gate on backend.
    is_native_vulkan = bool(getattr(profile, "native_vulkan", False))
    if res.numeric_ok and run_numeric and not has_weights and not is_native_vulkan:
        bad = _multishape_check(oracle, profile, cpp_path, params, model_py,
                                extra_sources, extra_includes, backend_kwargs, tol,
                                in0_squeezed=in0_squeezed)
        if bad is not None:
            res.numeric_ok = False
            res.failure_category = bad[0]
            res.numeric_log = bad[1]
            res.messages.append("multi-shape: failed a size variant")

    # ---- device-in-the-loop gate ----------------------------------------------
    # After the HOST verdict passes, verify on the REAL phone (correctness + latency).
    # A device fail (NDK compile / numeric divergence / crash) sets device_status
    # ="failed" and puts the diagnostic in numeric_log so it flows into the existing
    # numeric_repair prompt. No device / flaky -> "skipped" (keep the host result).
    # backend=="base"/"arm" -> DeviceOracle (CPU); "vulkan" -> VulkanDeviceOracle (Adreno).
    if device_verify != "off" and res.host_ok:
        try:
            from layer_oracle import DeviceOracle, VulkanDeviceOracle
            if profile.backend == "vulkan":
                dev = VulkanDeviceOracle(ncnn_root=getattr(oracle, "ncnn_root", None))
                dev_kwargs = {"shader": backend_kwargs.get("shader"),
                              "extra_shaders": backend_kwargs.get("extra_shaders", [])}
            else:
                dev = DeviceOracle(ncnn_root=getattr(oracle, "ncnn_root", None))
                dev_kwargs = {"packing": int(backend_kwargs.get("packing", 0))}
            avail, why = dev.available()
            if not avail:
                res.device_status = "skipped"
                res.messages.append(f"device gate skipped ({why})")
                if device_verify == "on":
                    res.messages.append("WARNING: --device-verify on but no device — kept host result")
            else:
                dv = dev.verify(
                    candidate_cpp=cpp_path, class_name=profile.class_name,
                    header=profile.header, params=params, inputs=ncnn_inputs,
                    reference=reference, weights=weights, weight_flags=weight_flags,
                    tol=tol, extra_sources=extra_sources, extra_includes=extra_includes,
                    bench=20, simpleperf=device_simpleperf, backend=profile.backend,
                    **dev_kwargs)
                if getattr(dv, "skipped", False):
                    res.device_status = "skipped"
                    res.messages.append(f"device gate skipped ({dv.detail})")
                elif dv.passed:
                    res.device_status = "passed"
                    res.device_latency = dv.latency
                    res.messages.append(f"device passed (max_diff={dv.max_diff}, "
                                        f"latency_min={dv.latency}ms)")
                else:
                    res.device_status = "failed"
                    res.failure_category = dv.failure_category or "device"
                    res.numeric_log = dv.detail   # -> numeric_repair feedback next round
                    res.messages.append(f"device FAILED ({dv.failure_category})")
        except Exception as exc:  # noqa: BLE001 — device gate must never break the host loop
            res.device_status = "skipped"
            res.messages.append(f"device gate error (skipped): {exc}")
    return res
