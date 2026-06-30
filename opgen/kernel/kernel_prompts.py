"""Role prompts for the KernelAgent: analyzer / coder / debugger.

The agent writes a from-scratch ncnn base (non-optimized) Layer kernel and
verifies it against PyTorch via LayerOracle.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kernel_schemas import KernelProfile


NCNN_LAYER_BACKGROUND = """\
You write a NEW, self-contained ncnn base (CPU, non-optimized) Layer kernel.
It will be compiled standalone and linked against libncnn.a (NOT added to the
ncnn source tree), instantiated directly via `new <Class>()`, and run with all
optimizations OFF (no packing, no fp16). So write a plain, correct forward.

Layer skeleton (subclass ncnn::Layer):
```cpp
// <file>.h
#ifndef CAND_X_H
#define CAND_X_H
#include "layer.h"
namespace ncnn {
class <Class> : public Layer {
public:
    <Class>();
    virtual int load_param(const ParamDict& pd);   // only if there are params
    virtual int load_model(const ModelBin& mb);     // only if there are weights
    virtual int forward(const Mat& bottom, Mat& top, const Option& opt) const;        // one_blob_only && !inplace
    // or: virtual int forward_inplace(Mat& bottom_top, const Option& opt) const;     // one_blob_only && inplace
    // or: virtual int forward(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const; // !one_blob_only
public:
    // params / Mat weight_data; ...
};
} // namespace ncnn
#endif

// <file>.cpp
#include "<file>.h"
#include <math.h>
namespace ncnn {
<Class>::<Class>() { one_blob_only = true; support_inplace = false; }
int <Class>::load_param(const ParamDict& pd) { /* p = pd.get(id, default); */ return 0; }
int <Class>::load_model(const ModelBin& mb) { weight = mb.load(N, 1); if (weight.empty()) return -100; return 0; }
int <Class>::forward(const Mat& bottom, Mat& top, const Option& opt) const { /* ... */ return 0; }
DEFINE_LAYER_CREATOR(<Class>)
} // namespace ncnn
```

forward interface by (one_blob_only, support_inplace):
  (true,false)->forward(const Mat&, Mat&, opt)   (true,true)->forward_inplace(Mat&, opt)
  (false,*)  ->forward(const std::vector<Mat>&, std::vector<Mat>&, opt)

ncnn::Mat layout (CRITICAL for correct indexing):
- dims/w/h/d/c: 1D=(w); 2D=(w,h); 3D=(w,h,c); 4D=(w,h,d,c). Per-channel pointer: `mat.channel(q)`
  gives the q-th channel base; elements within a channel are w*h*d contiguous; channels are
  separated by `mat.cstep` (NOT necessarily w*h*d — use channel(q), don't compute c*w*h yourself).
- allocate output: `top.create(w, h, c, elemsize, opt.blob_allocator); if (top.empty()) return -100;`
- read float: `const float* p = bottom.channel(q);`  write: `float* o = top.channel(q);`
- iterate: `for (int q=0;q<c;q++){ const float* p=bottom.channel(q); float* o=top.channel(q);
  for (int i=0;i<w*h;i++) o[i]=f(p[i]); }`

INPUT LAYOUT from the harness mirrors PyTorch with the batch dim dropped:
  torch (N,C,H,W)->ncnn Mat(w=W,h=H,c=C); (N,C,L)->Mat(w=L,h=C)?? NO: 2D torch (N,C)->Mat(w=C);
  3D torch (N,C,L)->Mat(w=L,h=C); 4D torch (N,C,H,W)->Mat(w=W,h=H,c=C).
  (The harness drops axis 0 and stores [ndim][dims][data]; ncnn rebuilds Mat with w=last dim.)
Match your forward's interpretation to this.

WEIGHTS: the harness passes weights as flat float arrays in the EXACT order of
profile.weight_keys (each = a PyTorch state_dict tensor flattened). Your load_model
must `mb.load(...)` them in the SAME order; index them in forward consistently with
PyTorch's tensor layout (e.g. conv weight is [out][in][kh][kw] row-major).

=== ncnn::Mat MEMORY MODEL — cstep / channel gap (READ THIS, IT IS THE #1 BUG SOURCE) ===
- `mat.cstep` is the stride between consecutive channels in floats, NOT necessarily
  `w*h*d`. ncnn pads each channel slice up to an alignment boundary so that
  `cstep >= w*h*d`, with the gap zero-filled. Concretely: when packing is OFF
  (elempack=1), cstep is `w*h*d` rounded up to a multiple of 4 — if `w*h*d` is not
  divisible by 4 there is a per-channel gap of unused floats between channels.
- THEREFORE: NEVER cast a Mat to a flat float* and iterate `w*h*d*c` elements:
      const float* p = (const float*)bottom;  // WRONG — will read gap garbage / write past
      for (int i = 0; i < w*h*d*c; i++) p[i] = ...;
  This silently corrupts channel boundaries whenever cstep != w*h*d. ALWAYS iterate
  per-channel via `mat.channel(q)`, processing `w*h*d` elements inside each channel:
      for (int q = 0; q < c; q++) {
          const float* pin  = bottom.channel(q);
          float*       pout = top.channel(q);
          for (int i = 0; i < w*h*d; i++) pout[i] = f(pin[i]);
      }
- 1D/2D Mats have no channel dim. Use `.dims` to branch (dims==1 → just w; dims==2 →
  w*h; dims==3 → loop over c using channel(q); dims==4 → loop over c, each channel
  holds w*h*d). Do NOT assume 4D.

=== PACKED LAYOUT — what changes when packing is ON (elempack > 1) ===
The base kernel itself runs at elempack=1 (no packing). But ncnn's packing path is
ALSO exercised against your kernel in regression checks, and an arm subclass WILL
run packed. Internal mental model so you don't write code that breaks under packing:
- When elempack=N, each "channel slot" stores N original channels interleaved at the
  innermost layout. `mat.c` becomes `original_c / elempack` (the number of channel
  *groups*), and `mat.elempack = N`. Each `channel(q)` then holds
  `w*h*d*elempack` contiguous floats: N values per spatial position, lane-interleaved.
- `mat.cstep` (in floats) still includes the per-channel-group alignment gap. If you
  write per-channel loops correctly (per the section above), the packed path "just
  works" for pure elementwise ops — process `w*h*d*elempack` floats per channel slot.
- Reductions across channels, broadcasting along channel, or anything that mixes lanes
  must explicitly handle elempack: either unpack first, or iterate
  `(q, lane)` with the right stride.
"""

ARM_LAYER_BACKGROUND = """\
THIS IS AN ARM (NEON) BACKEND KERNEL — it SUBCLASSES the base layer (which is
already written & verified). It is compiled together with the base .cpp and run
with PACKING ON (NC4HW4, elempack=4). Be numerically identical to the base op.

Conventions (CRITICAL):
- Header: `#include "{base_header}"` then `class {arm_class} : public {base_class}`.
  Override ONLY the forward method(s); inherit params/weights/flags from the base
  constructor by calling nothing special — the base ctor already ran.
- MANDATORY OVERRIDE: you MUST define `{arm_class}::forward` (or
  `{arm_class}::forward_inplace`) out-of-line in the .cpp with the NEON path. If you
  omit it, C++ silently dispatches to the inherited base CPU forward — the kernel is
  then a degraded base kernel (not arm) and WILL BE REJECTED even if numerics match.
- Constructor: enable packing:
    {arm_class}::{arm_class}() {{
    #if __ARM_NEON
        support_packing = true;
    #endif
    }}
- NEON: `#if __ARM_NEON` + `#include <arm_neon.h>`. Optional ncnn helpers (on the
  include path): `#include "neon_mathfun.h"` (exp_ps/log_ps/sin_ps/tanh_ps/...),
  `#include "arm_usability.h"`.
- PACKED layout: read `int elempack = bottom_top_blob.elempack;` (it is 4). Each
  `channel(q)` holds `w*h*d*elempack` contiguous floats. For an elementwise op,
  packing is transparent — just process those floats, vectorizing 4-at-a-time:
    for (int q=0;q<channels;q++) {{ float* p = blob.channel(q);
        int size = w*h*d*elempack, i=0;
        for (; i+4<=size; i+=4) {{ float32x4_t v=vld1q_f32(p+i); /* f(v) */ vst1q_f32(p+i,v); }}
        for (; i<size; i++) p[i] = /* f(p[i]) */; }}
  Keep a correct scalar tail; also stay correct if elempack==1 (fallback path).
- Allocate output with the SAME elempack & elemsize as input; for an inplace op,
  do not reallocate.
- Parallelize channels: `#pragma omp parallel for num_threads(opt.num_threads)`.
- Do NOT write DEFINE_LAYER_CREATOR — arm registration is automatic via cmake.
"""


BROADCASTING_PRIMER = """\
=== ncnn BROADCASTING RULES (this op takes MULTIPLE inputs — read this) ===
ncnn's broadcasting is INNER-AXIS-FIRST and goes in the OPPOSITE direction of
numpy / PyTorch (which right-aligns shapes). Get this wrong and your shape
contract silently picks the wrong axis.

Notation: shapes are written innermost-first as [w], [w,h], [w,h,c], [w,h,d,c]
(matching how Mat stores them: dims=1→w, dims=2→w+h, dims=3→w+h+c, dims=4→w+h+d+c).

1) SCALAR / SCALAR-LIKE — B is a singleton:
     A=[2,3,4]   B=scalar or [1] or [1,1] or [1,1,1]   → C=[2,3,4]
   Apply B's single value to every element of A.

2) SAME-SHAPE — straight element-wise, no broadcast:
     A=[2,3,4]   B=[2,3,4]   → C=[2,3,4]

3) EXPLICIT BROADCAST — B has matching rank but some axes are 1:
     A=[2,3,4,5]  B=[2,3,1,5]  → C=[2,3,4,5]   (B repeats along d=4)
     A=[2,3,4]    B=[1,3,1]    → C=[2,3,4]     (B repeats along w and c)
   For every axis where B==1, repeat B along that axis. Where B>1, axes must equal A.

4) IMPLICIT BROADCAST — B has LOWER rank than A. **Inner axis first**, opposite numpy:
     A=[2,3]      B=[3]        → C=[2,3]   (B is broadcast as [1,3], i.e. INNER w aligns)
     A=[2,3,4]    B=[4]        → C=[2,3,4] (B aligns to innermost w)
     A=[2,3,4]    B=[3,4]      → C=[2,3,4] (B aligns to [w,h], NOT [h,c])
     A=[2,3,4,5]  B=[3,4,5]    → C=[2,3,4,5] (B aligns to [w,h,d])
   Mental check: numpy would right-align as [3]→[1,1,3] for A=[2,3,4], broadcasting on
   the LAST axis. ncnn does the OPPOSITE: [3] aligns to A's [w], expanding outwards.

5) LEGACY 1-D OUTER-AXIS (only when (4) does NOT apply because sizes don't match
   the inner axis): B=[2] against A=[2,3] → C=[2,3]. If inner-axis match is possible
   it ALWAYS wins (see B=[2] vs A=[2,2] → inner-axis broadcast).

Implementation hints for your forward:
- Read `a.dims, a.w, a.h, a.d, a.c` and `b.dims, b.w, ...` first; branch on the
  combination instead of assuming a single shape. Most BinaryOp-style kernels need
  a small dispatch table over (a.dims, b.dims, equal/1-along-axis).
- If the model only ever feeds you one shape combo, you may special-case it — but at
  minimum print a clear error for unhandled combos (return -100) rather than read
  out-of-bounds.
- pnnx often inserts a `Reshape` upstream to convert implicit→explicit. Do not rely
  on that — your kernel still must handle implicit when the IR doesn't reshape.
"""


VULKAN_LAYER_BACKGROUND = """\
THIS IS A VULKAN (GPU) BACKEND KERNEL — it SUBCLASSES the base layer (already
written & verified). You author THREE files: a C++ header, a C++ source, and a
SEPARATE GLSL compute shader `{shader_file}` (the actual math lives in the shader).
It is verified by isolated instantiation on the GPU (`new {vulkan_class}()`, run
`forward` on VkMat) and must be numerically identical to the base op. v1 runs at
elempack=1 (the oracle force-unpacks inputs), so write a SCALAR shader.

Conventions (CRITICAL):
- Header `{vulkan_header}`: `#include "{base_header}"` then
  `class {vulkan_class} : public {base_class}`. Declare:
    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);
    using {base_class}::forward_inplace;   // if the op is inplace
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
    // or, if NOT inplace: virtual int forward(const VkMat& bottom, VkMat& top, VkCompute& cmd, const Option& opt) const;
  and a `Pipeline* pipeline_xxx;` member.
- Source `{vulkan_file}`: `#include "{vulkan_header}"` and
  `#include "cand_vulkan_shader.h"` (a helper on the include path; it reads the
  shader file and online-compiles it — see below).
- Constructor — MANDATORY: set `support_vulkan = true;` (also `support_inplace`
  as needed) and `pipeline_xxx = 0;`. If you omit `support_vulkan = true` the
  oracle REFUSES the kernel (it must not fall back to CPU).
- create_pipeline — compile the shader AT RUNTIME (do NOT reference
  `LayerShaderType::xxx`; that build-time enum is unavailable here):
    int {vulkan_class}::create_pipeline(const Option& opt) {{
        std::vector<uint32_t> spirv;
        if (compile_candidate_shader(opt, spirv) != 0) return -1;  // helper reads {shader_file}
        std::vector<vk_specialization_type> specializations(1);    // MUST match the shader's
        specializations[0].i = 0;                                  // constant_id count (1 here)
        pipeline_xxx = new Pipeline(vkdev);
        pipeline_xxx->set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1); // 1D!
        return pipeline_xxx->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
    }}
  NOTE the workgroup MUST be 1D `(subgroup_size,1,1)` to match a 1D dispatch — the
  default `set_optimal_local_size_xyz()` is 3D and leaves most elements unprocessed.
- destroy_pipeline: `delete pipeline_xxx; pipeline_xxx = 0; return 0;`
- forward_inplace(VkMat&, VkCompute& cmd, opt): dispatch over total elements:
    int n = (int)bottom_top_blob.total();   // elempack==1 -> scalar count
    std::vector<VkMat> bindings(1); bindings[0] = bottom_top_blob;
    std::vector<vk_constant_type> constants(1); constants[0].i = n;  // -> push_constant
    VkMat dispatcher; dispatcher.w = n; dispatcher.h = 1; dispatcher.c = 1;
    cmd.record_pipeline(pipeline_xxx, bindings, constants, dispatcher);
    return 0;
- Do NOT write DEFINE_LAYER_CREATOR (vulkan registration is automatic via cmake).

The shader `{shader_file}` — ncnn shader dialect, SCALAR (elempack=1):
```glsl
#version 450
layout(constant_id = 0) const int n = 0;
layout(binding = 0) buffer bottom_top_blob {{ sfp bottom_top_blob_data[]; }};
layout(push_constant) uniform parameter {{ int n; }} p;
void main() {{
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= psc(n)) return;
    afp v = buffer_ld1(bottom_top_blob_data, gi);
    v = /* f(v) — the op's math */;
    buffer_st1(bottom_top_blob_data, gi, v);
}}
```
Shader rules: `sfp`=storage float, `afp`=arithmetic float; load `buffer_ld1(buf,i)`,
store `buffer_st1(buf,i,v)`; `psc(x)` resolves a push-constant when its spec-const
is 0. Use ONLY scalar `sfp`/`buffer_ld1` (NOT `sfpvec4`/pack4) for v1. For multi-input
ops add more `layout(binding=k) readonly buffer ...` and more bindings in the .cpp.
"""


def _background(backend: str, profile: "KernelProfile | None" = None) -> str:
    # multi-input ops (one_blob_only == False) need the broadcasting primer,
    # regardless of backend
    bcast = (BROADCASTING_PRIMER if profile and not profile.one_blob_only else "")
    if backend not in ("arm", "vulkan"):
        return NCNN_LAYER_BACKGROUND + ("\n" + bcast if bcast else "")
    base_class = (profile.base_class if profile else "") or "BaseLayer"
    suffix = "_" + backend
    sub_class = (profile.class_name if profile else "") or (base_class + suffix)
    # derive base header from the base class name deterministically (cand_<x>.h)
    base_header = "base.h"
    if profile and profile.base_class:
        stem = (profile.base_class[len("Cand_"):].lower()
                if profile.base_class.startswith("Cand_") else profile.base_class.lower())
        base_header = f"cand_{stem}.h"
    if backend == "arm":
        addendum = ARM_LAYER_BACKGROUND.format(
            base_header=base_header, base_class=base_class, arm_class=sub_class)
    else:  # vulkan
        shader_file = (profile.shader if profile and profile.shader else "cand_x.comp")
        addendum = VULKAN_LAYER_BACKGROUND.format(
            base_header=base_header, base_class=base_class, vulkan_class=sub_class,
            vulkan_header=(profile.header if profile else "cand_x_vulkan.h"),
            vulkan_file=(profile.file if profile else "cand_x_vulkan.cpp"),
            shader_file=shader_file)
    return NCNN_LAYER_BACKGROUND + "\n" + addendum + ("\n" + bcast if bcast else "")


OUTPUT_CONTRACT = """\
OUTPUT CONTRACT — return ONLY fenced code blocks, nothing else.
The FIRST LINE INSIDE each fence MUST be the file name (header then source):
```cpp
<header.h>
// code
```
```cpp
<file.cpp>
// code
```
No prose outside the code blocks. Class name and #include must match the profile.
"""


def _examples(examples: dict[str, str]) -> str:
    if not examples:
        return "(no example retrieved)"
    return "\n\n".join(f"----- ncnn/src/layer/{k} -----\n{v}" for k, v in examples.items())


def _introspect(intro: dict | None) -> str:
    if not intro:
        return "(model introspection unavailable)"
    inp = intro.get("input_shapes") or []
    parts = [f"input shapes (torch, with batch): {inp}"]
    # If there are multiple inputs AND state_dict is empty, flag it as a
    # functional-op candidate (helps the LLM fill in weights_from_inputs).
    sd = intro.get("state_dict") or {}
    if len(inp) >= 2 and not sd:
        parts.append(f"  → NOTE: {len(inp)} inputs + empty state_dict suggests a "
                     "FUNCTIONAL op (weights arrive as forward inputs, not nn.Parameter). "
                     "See the FUNCTIONAL OPS section above and set weights_from_inputs "
                     "to the input indices that are actually weights.")
    if intro.get("output_shape"):
        nsh = intro.get("ncnn_output_shape") or []
        nelem = 1
        for d in nsh:
            nelem *= int(d)
        parts.append(f"EXPECTED OUTPUT (shape contract): torch {intro['output_shape']} -> ncnn Mat "
                     f"(batch dropped) {nsh} = {nelem} elements. Your forward MUST allocate top with "
                     f"exactly this shape/element-count.")
    if intro.get("state_dict"):
        parts.append("state_dict (key: shape) — these are the candidate weights:")
        for k, s in intro["state_dict"].items():
            parts.append(f"  {k}: {s}")
    else:
        parts.append("state_dict: (none — weightless op)")
    if intro.get("init_inputs") is not None:
        parts.append(f"get_init_inputs(): {intro['init_inputs']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
_MULTI_INPUT_ANALOGS = {"BinaryOp", "Eltwise", "Concat", "MatMul", "Gemm"}


def _interface_reference_block(task_name: str) -> str:
    """Inject the ncnn built-in layer interface for this task, when known.

    Returns an empty string for tasks whose analog ncnn layer can't be guessed,
    or whose layer isn't in the dictionary. The KernelAgent stays free to make
    its own decisions in that case (no regression on novel ops).

    For multi-input analogs (BinaryOp/Eltwise/Concat/MatMul/Gemm) the
    broadcasting primer is also appended so the ANALYZER prompt itself sees
    ncnn's inner-axis-first rules — important because the analyzer is what
    decides one_blob_only=false and rank_coverage.
    """
    # late import keeps this lookup optional: tests / standalone callers that
    # don't bootstrap_paths still work, they just won't get the reference block.
    try:
        from lookup import guess_layer_from_task, render_for_prompt
    except ImportError:
        return ""
    layer = guess_layer_from_task(task_name)
    if not layer:
        return ""
    block = render_for_prompt(layer, role="kernel")
    if not block:
        return ""
    suffix = (
        f"\n{block}\n"
        f"NOTE: the above is the EXACT interface of the corresponding ncnn "
        f"built-in layer. Your `params` keys and `weight_keys` order MUST follow it. "
        f"If your op truly needs a different interface, set `analog_layer` to "
        f"something other than `{layer}` to opt out.\n"
    )
    if layer in _MULTI_INPUT_ANALOGS:
        suffix += "\n" + BROADCASTING_PRIMER
    return suffix


_FUNCTIONAL_OP_GUIDE = """\
=== FUNCTIONAL OPS — when weights arrive as INPUTS, not state_dict ===
Some operators are implemented with `torch.nn.functional` (F.conv2d, F.linear,
F.layer_norm, F.batch_norm, ...) where the weight/bias tensors are passed in via
`forward(x, weight, bias)` rather than stored as `nn.Parameter` on the module.
Symptoms in the introspection above:
  - `state_dict: (none — weightless op)`  AND  `input shapes` shows MORE than one
    tensor (the first is the activation; the rest are weight-like tensors).

For these ops, your kernel's INPUTS to the ncnn forward are still ONLY the
activation(s) — the weight tensors are routed to your `load_model(ModelBin&)`
exactly as for a normal nn.Module-backed layer. The harness handles the
splitting automatically when you tell it via `weights_from_inputs`:

  - `weights_from_inputs: []`                — no functional weights (the default)
  - `weights_from_inputs: [1, 2]`            — input[1] and input[2] are weights;
    they will be removed from the bottom_blobs list and passed to mb.load in
    THIS order. input[0] (and any other indices not listed) stay as activations.
  - `weight_keys` should be `[]` in this case (state_dict is empty — there are no
    keys to name). Don't write `["weight","bias"]` here, that will only error.

Concrete mapping for `F.conv2d(x, w, b)` with inputs `[x, w, b]`:
  - weights_from_inputs: [1, 2]   (w then b → mb.load order = [w, b])
  - weight_keys:         []
  - load_model: `weight = mb.load(weight_data_size, 0); bias = mb.load(num_output, 1);`
  - forward sees ONE bottom blob (the activation x) and uses self.weight / self.bias.

When `state_dict` is non-empty and the model is an nn.Module wrapper, stick with
`weight_keys` (the existing path). Don't set `weights_from_inputs` in that case.
"""


def analyzer_prompt(task_name: str, model_code: str, intro: dict | None) -> str:
    ref = _interface_reference_block(task_name)
    return f"""Analyze a PyTorch operator and plan a from-scratch ncnn base kernel.

Task: {task_name}
PyTorch model:
```python
{model_code}
```

Model introspection (ground truth):
{_introspect(intro)}

{NCNN_LAYER_BACKGROUND}

{_FUNCTIONAL_OP_GUIDE}
{ref}
Return ONLY a JSON object (no prose). FIELDS MUST BE SIMPLE VALUES:
- `class_name`, `header`, `file`: short identifiers / filenames ONLY (no code,
  no `#include`, no braces, no newlines). e.g. `Cand_Abs` / `cand_abs.h`.
- The actual .h/.cpp source code is emitted later in the coder phase — NOT here.

{{
  "class_name": "Cand_{task_name}",
  "header": "cand_{task_name.lower()}.h",
  "file": "cand_{task_name.lower()}.cpp",
  "one_blob_only": true,
  "support_inplace": false,
  "params": {{}},                      // {{param_id: concrete value for THIS model}}, e.g. {{"0": 1.0}}
  "weight_keys": [],                   // state_dict keys in load_model order (e.g. ["weight","bias"]); [] if weightless OR functional (use weights_from_inputs instead)
  "weights_from_inputs": [],           // input indices that are weights (e.g. [1, 2] for F.conv2d(x, w, b)); [] for nn.Module-backed ops
  "analog_layer": "<nearest existing ncnn base layer file stem, e.g. absval / elu / convolution>",
  "notes": "<op math + how the forward should index the Mat / weights>"
}}"""


def parse_profile_json(task_name: str, text: str, backend: str = "base") -> KernelProfile:
    payload: dict[str, Any] = {}
    for blk in reversed(re.findall(r"\{.*\}", text, re.DOTALL)):
        try:
            payload = json.loads(blk)
            break
        except json.JSONDecodeError:
            continue
    return KernelProfile.from_llm(task_name, payload, backend=backend)


def _files_instruction(profile: KernelProfile) -> str:
    """Backend-aware 'which files to emit' line for the coder/debugger prompts."""
    if profile.backend == "vulkan":
        return (f"Write exactly THREE files: {profile.header}, {profile.file}, and the "
                f"GLSL shader {profile.shader}. The class must be `{profile.class_name}`. "
                f"Do NOT write DEFINE_LAYER_CREATOR. Do NOT reference LayerShaderType.")
    if profile.backend == "arm":
        return (f"Write exactly two files: {profile.header} and {profile.file}. The class "
                f"must be `{profile.class_name}` and MUST override forward/forward_inplace. "
                f"Do NOT write DEFINE_LAYER_CREATOR (arm registration is automatic).")
    return (f"Write exactly two files: {profile.header} and {profile.file}. The class must be "
            f"`{profile.class_name}` and end the .cpp with DEFINE_LAYER_CREATOR({profile.class_name}).")


def _functional_routing_note(profile: KernelProfile, intro: dict | None) -> str:
    """If the op's weights come from forward INPUTS (functional style), tell the
    coder exactly which bottom_blob index becomes which mb.load slot.
    """
    wfi = getattr(profile, "weights_from_inputs", None) or []
    if not wfi:
        return ""
    in_shapes = (intro or {}).get("input_shapes") or []
    activation_idx = [i for i in range(len(in_shapes)) if i not in wfi]
    parts = [
        "=== FUNCTIONAL OP — input-vs-weight routing ===",
        f"This op's weights are routed from forward inputs at indices {wfi} "
        "(in load_model order). The harness will:",
        f"  - pass input[{activation_idx}] to your forward as bottom blobs",
        f"  - pass input[{wfi}] to your load_model via mb.load(..., flag=k)",
    ]
    for k, src_idx in enumerate(wfi):
        if src_idx < len(in_shapes):
            parts.append(f"      slot {k}  ← input[{src_idx}], shape={in_shapes[src_idx]}")
    parts.append("Your kernel should: declare Mat members for these weights, "
                 "mb.load them in order in load_model, and reference them in "
                 "forward — NEVER expect them as bottom_blobs.")
    return "\n".join(parts) + "\n"


def coder_prompt(profile: KernelProfile, examples: dict[str, str], model_code: str, intro: dict | None) -> str:
    return f"""Write the from-scratch ncnn {profile.backend} kernel for this operator.

{_background(profile.backend, profile)}

{_functional_routing_note(profile, intro)}
Kernel profile (follow it exactly — class name, params ids, weight order):
{json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)}

Model introspection (ground truth shapes / weights):
{_introspect(intro)}

PyTorch model (the semantics to reproduce):
```python
{model_code}
```

Nearest existing ncnn base layer(s) to imitate (style + Mat usage):
{_examples(examples)}

{_files_instruction(profile)}

{OUTPUT_CONTRACT}"""


_PHASE = {
    "generate_repair": "No valid .h/.cpp blocks were produced.",
    "compile_repair": "The kernel failed to COMPILE.",
    "numeric_repair": "The kernel compiled and ran but its output does NOT match PyTorch "
                      "(allclose failed) or crashed/has wrong shape.",
}


def debugger_prompt(phase: str, profile: KernelProfile, code_book: dict[str, str],
                    feedback: str, memory: str, intro: dict | None) -> str:
    framing = _PHASE.get(phase, "The kernel failed.")
    cur = "\n\n".join(f"----- {p} -----\n{c}" for p, c in code_book.items()) or "(none)"
    extra = ""
    if phase == "numeric_repair":
        extra = ("\nCommon causes: wrong Mat indexing (use channel(q), not c*w*h), wrong axis/"
                 "shape, wrong weight layout/order vs PyTorch, missing activation/eps, "
                 "uninitialized output, off-by-one in loops.")
        if profile.backend == "arm":
            extra += ("\nARM-specific: forgot elempack (size must be w*h*d*elempack), wrong NEON "
                      "tail handling, mismatched output elempack/elemsize, or computing a different "
                      "value than the base op. Must match PyTorch after unpacking to elempack=1.")
        if profile.backend == "vulkan":
            extra += ("\nVULKAN-specific: forgot `support_vulkan = true` (rejected); 3D workgroup "
                      "instead of 1D `(subgroup_size,1,1)` (only part of the data processed); "
                      "specialization vector length != shader constant_id count; referenced "
                      "LayerShaderType (unavailable — use compile_candidate_shader); used "
                      "sfpvec4/pack4 instead of scalar sfp/buffer_ld1 at elempack=1; wrong push "
                      "constant wiring (psc(n) reads p.n). Must match PyTorch.")
    return f"""Repair the from-scratch ncnn {profile.backend} kernel.

Situation: {framing}{extra}

{_background(profile.backend, profile)}

{_functional_routing_note(profile, intro)}
Profile:
{json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)}

Model introspection:
{_introspect(intro)}

Current files:
{cur}

Diagnostic feedback:
```
{feedback}
```

History:
{memory or "(none)"}

{_files_instruction(profile)}

State the root cause in 1-3 sentences, then return the COMPLETE corrected files.

{OUTPUT_CONTRACT}"""
