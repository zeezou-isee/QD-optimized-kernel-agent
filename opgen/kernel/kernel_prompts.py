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
int <Class>::load_model(const ModelBin& mb) { weight = mb.load(N, 0); /*primary:type 0*/ bias = mb.load(M, 1); /*secondary:type 1*/ if (weight.empty()) return -100; return 0; }
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

mb.load(size, TYPE) — the TYPE is NOT free choice; it MUST match how ncnn stored
that weight, or the reader misaligns by 4 bytes and every value is garbage (looks
like a random sign-flipped result, ~99% wrong). Rule (from the REFERENCE interface
block below — use its `flag` field as the TYPE):
  - flag=0  → PRIMARY weight (ncnn fwrite_weight_tag_data: a 4-byte tag precedes
              the data).  Read with `mb.load(size, 0)`.  This is the layer's main
              weight: Convolution/InnerProduct/Gemm `weight_data`/`A`/`B`/`C`.
  - flag=1  → SECONDARY weight (ncnn fwrite_weight_data: raw, NO tag).  Read with
              `mb.load(size, 1)`.  This is bias_data AND every weight of layers
              like BatchNorm (slope/mean/var/bias all flag=1), Scale, PReLU.
If the REFERENCE block lists a weight with flag=0, you MUST use type 0 for it even
though it feels unusual; using type 1 there is the single most common e2e/oracle
failure. When in doubt, mirror the built-in layer's own load_model exactly.

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

=== PACKING IS OFF in this harness (elempack == 1) ===
Both LayerOracle and the real ncnn::Net (NetOracle / production) run with
`opt.use_packing_layout = false`. Every Mat you get has `elempack == 1`,
`elemsize == sizeof(float)` — plain row-major fp32. You do NOT need any
elempack>1 (NC4HW4) packed path, not even in the arm subclass. Write correct
elempack=1 code; on arm, vectorise 4-wide over the contiguous axis (width / the
flat w*h*d run per channel), not over packed lanes.
(Background only, not required here: if packing were ON, elempack=N would
interleave N channels per slot with `mat.c = original_c/elempack` and each
`channel(q)` holding `w*h*d*elempack` lane-interleaved floats — a future
fp16+packing validation pass would exercise that. It is OFF now.)
"""

ARM_LAYER_BACKGROUND = """\
THIS IS AN ARM (NEON) BACKEND KERNEL — it SUBCLASSES the base layer (which is
already written & verified). It is compiled together with the base .cpp. Be
numerically identical to the base op.

PACKING IS OFF — elempack == 1 (CRITICAL, read this first):
  The validator (LayerOracle) AND the real ncnn::Net (NetOracle / production) both
  run with `opt.use_packing_layout = false`. So EVERY Mat your forward receives has
  `elempack == 1` and `elemsize == sizeof(float)` — plain row-major fp32, exactly
  like the base kernel. You do NOT need to handle the NC4HW4 (elempack=4) packed
  layout; getting packed broadcast right is the #1 cause of arm failures and it is
  a path that is never exercised here. Your NEON win comes from vectorising 4-wide
  over the CONTIGUOUS axis (width / the flat w*h*d run per channel), NOT from
  elempack lanes.
  - DO NOT branch on elempack or read `bottom.elempack` to pick lanes; treat it as 1.
  - DO NOT set `support_packing = true` (that would invite elempack=4 input in real
    packed deployment which this kernel does not handle). Leave it false (the base
    default) so ncnn guarantees elempack=1 input.
  - Allocate output with `elemsize=sizeof(float)` and the same shape as the base.

Conventions (CRITICAL):
- Header: `#include "{base_header}"` then `class {arm_class} : public {base_class}`.
  Override ONLY the forward method(s); inherit params/weights/flags from the base
  constructor — the base ctor already ran.
- MANDATORY OVERRIDE: you MUST define `{arm_class}::forward` (or
  `{arm_class}::forward_inplace`) out-of-line in the .cpp with the NEON path. If you
  omit it, C++ silently dispatches to the inherited base CPU forward — the kernel is
  then a degraded base kernel (not arm) and WILL BE REJECTED even if numerics match.
- NEON: `#if __ARM_NEON` + `#include <arm_neon.h>`. Optional ncnn helpers (on the
  include path): `#include "neon_mathfun.h"` (exp_ps/log_ps/sin_ps/tanh_ps/...),
  `#include "arm_usability.h"`. Guard the NEON body with `#if __ARM_NEON` and keep a
  correct plain-C fallback for the non-NEON build.
- Vectorise 4-wide over the contiguous run, with a correct scalar tail:
    for (int q=0;q<channels;q++) {{ const float* p = bottom.channel(q); float* o = top.channel(q);
        int size = w*h*d, i=0;                     // elempack==1, so size is the real count
        for (; i+4<=size; i+=4) {{ float32x4_t v=vld1q_f32(p+i); /* f(v) */ vst1q_f32(o+i,v); }}
        for (; i<size; i++) o[i] = /* f(p[i]) */; }}
  Iterate per channel via `channel(q)` (respect cstep); never flat-index across channels.
- Parallelize channels: `#pragma omp parallel for num_threads(opt.num_threads)`.
- Do NOT write DEFINE_LAYER_CREATOR — arm registration is automatic via cmake.
"""


ARM_FUNCTIONAL_STRATEGY = """\
=== ARM STRATEGY FOR FUNCTIONAL OPS (THIS OP — read this carefully) ===
This is a FUNCTIONAL op (weight is bottom_blobs[1..], not pre-loaded). Standard
ncnn arm layers (Convolution_arm etc.) DO ZERO arm optimization in this case —
they `create_layer_cpu(LayerType::Convolution)` and dispatch to base CPU. That
behavior is the BASELINE you must beat. You CAN do better, because most arm
gains DO survive — weight-prep amortisation does not. Here is the rulebook:

CRITICAL FACTS THE HARNESS GUARANTEES (read these BEFORE writing any code):

  G1) ALL bottom_blobs have elempack == 1.
      The harness forces `_packing=0` for arm+functional. The runner does NOT
      call `convert_packing(...)` on any of your inputs. So:
        - `bottom_blob.elempack == 1` (activation, weight, bias — all of them)
        - DO NOT branch on elempack; DO NOT preserve `out_elempack = elempack`.
        - Allocate top_blob with `elempack=1` and `elemsize=sizeof(float)`:
            top_blob.create(out_w, out_h, num_output, sizeof(float), opt.blob_allocator);
        - Your NEON `vld1q_f32 / vst1q_f32` still works — you are 4-wide
          vectorising over OUTPUT WIDTH (the contiguous w axis), NOT over
          interleaved elempack lanes.

  G2) Weight Mat layout — ncnn axis order is REVERSED from PyTorch.
      The harness writes weight to a .bin in PyTorch order [out, in, kh, kw],
      then `Mat::create(dims[3], dims[2], dims[1], dims[0])` rebuilds it as
      a 4D ncnn Mat with **the REVERSE of that on each axis**:
        ncnn Mat axes after create:  w=kw,  h=kh,  d=in_ch,  c=out_ch
        So weight_blob.c == num_output, weight_blob.d == in_channels_per_group,
        weight_blob.h == kernel_h, weight_blob.w == kernel_w.
      Correct element access for `weight[out_ch][in_ch][kh][kw]` is:
        const float* p = weight_blob.channel(out_ch)   // 3D Mat (d=in, h=kh, w=kw)
                                    .depth(in_ch)       // 2D Mat (h=kh, w=kw)
                                    .row(kh);           // float* of length kw
        float v = p[kw];
      Hoist the .channel(out_ch).depth(in_ch) Mats OUTSIDE the spatial loop —
      re-resolving per element kills cache. NEVER cast weight to a flat
      `(const float*)weight_blob` then index it as `weight_ptr[oc*in*kh*kw + ...]`
      — `cstep` may be padded; channel-by-channel access is the only safe way.

  G3) Bias Mat layout — 1D, NOT a channel slice.
      bias_blob has dims=1, w=num_output, h=1, d=1, c=1. So:
        - `bias_blob.channel(0)` is WRONG (returns a 0-dim sub-Mat; bias[c] is OOB)
        - Correct: `const float* bias = (const float*)bias_blob;` (or `bias_blob.row(0)`)
        - Then `bias[oc]` indexes the per-output-channel bias scalar.

  G4) Activation Mat layout — 3D, ncnn's standard NCHW-minus-batch.
      bottom_blobs[0] has dims=3 (because the harness already dropped the
      torch batch axis): c=in_channels, h=H, w=W. Use channel(in_ch).row(y)[x]
      for activation pixels.

      (Param ID rules — see "PARAM IDs ARE NCNN, NOT ONNX" section below; it
      applies to BOTH base and arm functional kernels, not just arm.)

YOU MUST DO (these win net, even with single-forward amortisation):
  1) NEON 4-wide vector inner loop. The innermost contiguous dimension (usually
     output width W) is the loop to vectorize. Use `float32x4_t`, `vmlaq_f32`,
     `vld1q_f32`, `vst1q_f32`, etc. Scalar tail handler for `W % 4 != 0`.
  2) `#pragma omp parallel for num_threads(opt.num_threads)` over the OUTERMOST
     loop you can without false sharing — channel-out for conv, row for matmul.
  3) Kernel-size SPECIALIZATION via `if` ladder when possible:
       if (kernel_w == 1 && kernel_h == 1) {{ /* 1x1 fast path */ }}
       else if (kernel_w == 3 && kernel_h == 3 && stride_w == 1) {{ /* 3x3s1 */ }}
       else {{ /* generic */ }}
     The params are already in ParamDict at load_param time — branch is FREE.
  4) Hoist `bottom_blobs[1].channel(...)` pointers out of the inner loop —
     re-resolving the Mat indexer per element kills cache locality.
  5) `__builtin_prefetch(weight_ptr + 64)` before inner loop iterations on long
     reductions (conv channel-in sum). Free on ARM.

YOU MUST NOT DO (these need weight pre-processing → fail to amortise in a
single forward, will make you SLOWER than the base CPU baseline):
  ✗ Winograd input/weight transform (`conv3x3s1_winograd23/43/63`). The weight
    transform alone is O(num_output*num_input*36) and only pays off across
    many forwards. Do NOT call any helper named `*_winograd_*`.
  ✗ NC4HW4 weight pre-pack (`convolution_transform_kernel_packed_neon` and
    cousins). The repack cost is not amortised; just iterate weight in its
    raw PyTorch layout [out, in, kh, kw] inside the loop.
  ✗ im2col → cblas GEMM. The im2col buffer setup + memcpy overshoot single-
    forward break-even for small models.
  ✗ int8 / fp16 quantization of weight at runtime. The scale-calibration
    cost dwarfs anything you save.
  ✗ Touching the runtime weight tensor's elempack/elemsize. The
    `_packing=0` setting in the harness keeps it raw — KEEP IT THAT WAY.

FORWARD SIGNATURE (functional → multi-input layer, see profile.one_blob_only):
    int {arm_class}::forward(const std::vector<Mat>& bottom_blobs,
                              std::vector<Mat>& top_blobs,
                              const Option& opt) const
- bottom_blobs[0] is the activation; bottom_blobs[1..] are the runtime weights
  (the per-slot semantics are in `_functional_routing_note`).
- load_model is EMPTY (`return 0;`) — there is no pre-loaded weight to find.
- create_pipeline / destroy_pipeline do NOTHING (do not override them).

WHY THIS DESIGN (so you can judge edge cases):
- Static-weight conv pays the weight transform cost ONCE at load_model, then
  reuses the transformed weight across thousands of forwards. Net win.
- Functional conv would pay the transform on EVERY forward — the math says
  break-even is typically 5-50 forwards depending on conv shape. For our
  single-forward benchmark scenario the math says skip the transform.
- But the inner-loop arm gains (NEON 4-wide MAC, omp channel parallelism,
  branch-free kernel-size specialization) have ZERO per-forward setup cost.
  Those are pure wins. That is the gap you are filling that ncnn left open.
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


VULKAN_HOST_SIDE_BACKGROUND = """\
THIS IS A VULKAN (GPU) BACKEND KERNEL — it SUBCLASSES the base layer (already
written & verified). You author THREE files: a C++ header, a C++ source, and one
OR MORE GLSL compute shader files (the actual math lives in the shaders). It is
verified by isolated instantiation on the GPU (`new {vulkan_class}()`, run
`forward` on VkMat) and must be numerically identical to the base op. v1 runs at
elempack=1 (the oracle force-unpacks inputs to unpacked scalar Mat), so author
SCALAR shaders (sfp/buffer_ld1). If your op needs multiple pipelines (e.g. a
broadcast fast-path + a general path) write multiple .comp files and load each
via compile_candidate_shader_by_name.

Host-side conventions (CRITICAL):
- Header `{vulkan_header}`: `#include "{base_header}"` then
  `class {vulkan_class} : public {base_class}`. Declare:
    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);
    using {base_class}::forward_inplace;   // if the op is inplace
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
    // or, if NOT inplace: virtual int forward(const VkMat& bottom, VkMat& top, VkCompute& cmd, const Option& opt) const;
    // or, if multi-input: virtual int forward(const std::vector<VkMat>& bottom_blobs,
    //                                          std::vector<VkMat>& top_blobs,
    //                                          VkCompute& cmd, const Option& opt) const;
  and one `Pipeline* pipeline_xxx;` member PER shader you use.
- Source `{vulkan_file}`: `#include "{vulkan_header}"` and
  `#include "cand_vulkan_shader.h"` (a helper on the include path; it reads the
  .comp file(s) and online-compiles them — see below).
- Constructor — MANDATORY: set `support_vulkan = true;` (also `support_inplace`
  as needed) and every `pipeline_xxx = 0;`. Omit `support_vulkan = true` and the
  oracle REFUSES the kernel (it must not fall back to CPU).
- create_pipeline — compile shader(s) AT RUNTIME (do NOT reference
  `LayerShaderType::xxx`; that build-time enum is UNAVAILABLE here):
    int {vulkan_class}::create_pipeline(const Option& opt) {{
        std::vector<uint32_t> spirv;
        if (compile_candidate_shader(opt, spirv) != 0) return -1;   // primary shader
        std::vector<vk_specialization_type> specializations(N);     // spec-const count MUST
        specializations[0].i = ...;                                  // match the shader's
        //   ... one entry per `layout(constant_id = k) const T name = ...;`
        pipeline_xxx = new Pipeline(vkdev);
        pipeline_xxx->set_optimal_local_size_xyz(local_size_xyz);    // see per-op template
        int ret = pipeline_xxx->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations);
        if (ret != 0) return ret;
        // repeat for each extra shader:
        // if (compile_candidate_shader_by_name(opt, "myop_broadcast", spirv) != 0) return -1;
        // pipeline_broadcast->create(...);
        return 0;
    }}
- destroy_pipeline: `delete` every non-null pipeline_xxx, zero the pointer.
- forward*(): record ONE dispatch per pipeline you want to run. Dispatcher shape
  MUST match your workgroup / shader indexing (see per-op template).
- Do NOT write DEFINE_LAYER_CREATOR (vulkan registration is automatic via cmake).

Multi-input ops with weights (BinaryOp with a constant b, InnerProduct, ...):
weights are loaded by the BASE class via load_model(mb). Convert the base's CPU
`Mat weight_data` to a `VkMat weight_data_gpu` in `upload_model(cmd, opt)` — but
for v1 you can DELEGATE to the base's upload_model. If the base already provides
`upload_model`, override only when your GPU layout needs to differ.
"""


VULKAN_SHADER_DIALECT_BACKGROUND = """\
=== NCNN SHADER DIALECT MANUAL (READ FULLY BEFORE WRITING GLSL) ===

This is NOT plain GLSL — ncnn defines a small preprocessor + type macro layer
so the SAME shader source compiles for fp32 / fp16-storage / fp16-arithmetic
depending on the runtime option. If you write `float`/`vec4` directly your
shader may work at fp32 but crash / return garbage under fp16. USE THE MACROS.

1) TYPE MACROS
   - `sfp`  = storage float. ONE element in the buffer as it lives in VkMemory.
              (fp32 → float; fp16_storage → float16_t; you never care which.)
   - `afp`  = arithmetic float. What you compute with in registers.
              (fp32 → float; fp16_arithmetic → float16_t.)
   - `sfpvec4` / `afpvec4` = packed 4-lane variants for elempack=4 shaders.
   - v1 constraint: elempack=1 only. Use `sfp` / `afp` — NEVER `sfpvec4`.

2) LOAD/STORE MACROS
   - `afp v = buffer_ld1(buf, i);` load one element (fp16→fp32 conversion baked in)
   - `buffer_st1(buf, i, v);`      store one element
   - `buffer_ld4` / `buffer_st4`   for the vec4 variants (do NOT use in v1).
   - Element index `i` is in ELEMENT COUNT, not bytes. It counts sfp slots.

3) BUFFER BINDINGS
   - `layout(binding = 0) [readonly|writeonly] buffer name {{ sfp name_data[]; }};`
   - `readonly` for inputs, `writeonly` for outputs, unqualified for inplace.
   - The C++ side passes VkMats in the SAME order as bindings[0], bindings[1], ...
   - Bindings must be contiguous (0, 1, 2, ...) — a gap makes ncnn crash.

4) PUSH CONSTANTS + `psc(x)` — MOST COMMON SHADER-COMPILE FAIL
   - `layout(push_constant) uniform parameter {{ int w; int h; int c; int cstep; }} p;`
   - Access via `p.w` etc.
   - `psc(NAME)` is an ncnn MACRO. Its EXACT expansion is:
         `psc(x)  →  (x==0 ? p.x : x)`
     which means EVERY name you pass to psc() MUST be BOTH:
       (a) declared as a SPEC-CONST with default 0:
           `layout(constant_id = SHAPE_OFFSET + K) const int NAME = 0;`
       (b) declared as a PUSH_CONSTANT field:
           inside `layout(push_constant) uniform parameter {{ ... int NAME; ... }} p;`
     If EITHER is missing, glslang errors with `'NAME' : undeclared identifier`
     and the shader FAILS TO COMPILE (SHADER_COMPILE_FAIL rc=51).
   - Convention: pass shape hints as spec-consts when known at compile time
     (via `specializations[i].i = shape.x` in the host); pass RUNTIME shape via
     push_constants (via `constants[i].i = mat.x`) and read via `psc()`.
   - ALWAYS access shape via `psc(x)`, NEVER via plain `p.x`. Using `p.x`
     defeats the constant-folding fast path when the shape IS known.
   - CHECKLIST before submitting a shader: for every `psc(NAME)` call, grep
     the shader for both `constant_id = ... const int NAME = 0;` AND
     `int NAME;` inside push_constant. Both must be present.

5) SPECIALIZATION CONSTANTS
   - `layout(constant_id = 0) const int op_type = 0;` — set from the C++
     `specializations[0].i` slot; count MUST match ordinally in the C++ vector.
   - Use spec-consts for values that partition the shader (op_type, axis, dims).
     Use push_constants for values that change per dispatch (shape, cstep).
   - MISMATCH = pipeline_create rejects the pipeline (RC_PIPELINE_CREATE_FAIL).

6) NCNN MAT INDEXING (CRITICAL — most vulkan port bugs live here)
   ncnn Mats are NOT flat contiguous fp32 arrays. Each channel starts at an
   ALIGNED offset called `cstep`, which is USUALLY `w*h*d` but can be padded up
   for SIMD (e.g. rounded to a multiple of 4 or 8 elements). The right index:
       gi = c * cstep + h * w + x        // 3D Mat, one element
       gi = c * cstep + d * (h*w) + h * w + x   // 4D Mat
   NEVER use `c * w * h + h * w + w` — that's the WRONG index once cstep > w*h.
   For inputs: pass `p.cstep = bottom.cstep;` and index with `c * psc(cstep) + ...`.

7) DISPATCH SHAPE
   The runner's `dispatcher` VkMat controls how many workgroups are launched.
   The GLOBAL invocation count = `dispatcher.{{w,h,c}}` rounded UP to local_size.
   - 1D shader (elementwise): dispatcher.w = n; h=1; c=1. Workgroup 1D.
   - 3D shader (per-c/h/w): dispatcher.w = out.w; h = out.h; c = out.c. Workgroup 3D.
   Your shader `main()` reads `gl_GlobalInvocationID.{{x,y,z}}` — but each maps to
   `dispatcher.{{w,h,c}}` (X→w, Y→h, Z→c). The .x axis is NOT "cols" or "batch",
   it's the innermost/w axis. GUARD every axis: `if (gx >= psc(w)) return;`.

8) OUT-OF-BOUNDS GUARDS
   Every shader main() MUST early-return before any load/store when the global
   ID is past the actual shape:
       if (gx >= psc(w) || gy >= psc(h) || gz >= psc(c)) return;
   Missing this crashes MoltenVK/GPU driver → RC_DISPATCH_FAIL, not a soft error.

9) COMMON PITFALLS SEEN IN LLM-WRITTEN SHADERS
   - 3D workgroup with a 1D dispatcher → only x-lane runs, y/z go idle
     (or vice versa: 1D workgroup with 3D dispatcher wastes 63/64 threads).
   - Missed the cstep concept → all c>0 elements read wrong data.
   - Read a scalar push_constant via `p.n` instead of `psc(n)` → shape-hint
     branch mismatched with what create_pipeline compiled the shader for.
   - Wrote `float` / `vec4` instead of `sfp` / `sfpvec4` → fp16_storage crash.
   - Forgot `#version 450` at the top → glslang refuses to compile.
   - Bindings count in shader ≠ bindings vector size in .cpp → pipeline reject.
   - specializations vector length ≠ spec-const count in shader → pipeline reject.
"""


# Per-op-family templates. The KernelAgent picks one based on the profile's
# analog_layer + op traits (elementwise / broadcast / reduction / conv / gemm).
# Each template shows a WORKING skeleton the LLM can adapt — dispatcher shape,
# workgroup, bindings, push-constant layout — so the shader you get back has the
# right shape from turn 0 instead of iterating on the mechanics for 3 rounds.
_VULKAN_TEMPLATE_ELEMENTWISE = """\
=== PER-OP TEMPLATE: elementwise 1-in-1-out (Abs / ReLU / Sigmoid / ...) ===

Shader (elempack=1, 1D dispatch over total element count):
```glsl
#version 450
#define shape_constant_id_offset 0
layout(constant_id = shape_constant_id_offset + 0) const int n = 0;

layout(binding = 0) buffer bottom_top_blob {{ sfp bottom_top_blob_data[]; }};

layout(push_constant) uniform parameter {{ int n; }} p;

void main() {{
    const int gi = int(gl_GlobalInvocationID.x);
    if (gi >= psc(n)) return;
    afp v = buffer_ld1(bottom_top_blob_data, gi);
    v = /* f(v) — the op's math, e.g. abs(v), max(afp(0), v), 1.0/(1.0+exp(-v)) */;
    buffer_st1(bottom_top_blob_data, gi, v);
}}
```

Host create_pipeline: 1D workgroup, one spec-const, one push-constant.
```cpp
int Cand::create_pipeline(const Option& opt) {{
    std::vector<uint32_t> spirv;
    if (compile_candidate_shader(opt, spirv) != 0) return -1;
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = 0;                            // n unknown at compile → 0
    pipeline_xxx = new Pipeline(vkdev);
    pipeline_xxx->set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1);
    return pipeline_xxx->create(spirv.data(), spirv.size() * sizeof(uint32_t),
                                specializations);
}}
```

Host forward_inplace: 1D dispatcher over total().
```cpp
int Cand::forward_inplace(VkMat& b, VkCompute& cmd, const Option& opt) const {{
    int n = (int)b.total();
    std::vector<VkMat> bindings(1); bindings[0] = b;
    std::vector<vk_constant_type> constants(1); constants[0].i = n;
    VkMat dispatcher; dispatcher.w = n; dispatcher.h = 1; dispatcher.c = 1;
    cmd.record_pipeline(pipeline_xxx, bindings, constants, dispatcher);
    return 0;
}}
```
"""


_VULKAN_TEMPLATE_BINARYOP = """\
=== PER-OP TEMPLATE: two-input elementwise / broadcast (Add / Mul / Sub / Pow / ...) ===

For broadcast ops with two inputs of possibly-different shape, use a 3D shader
so you can address (c, h, w) independently.

Shader (elempack=1, 3D dispatch, broadcast-safe):
```glsl
#version 450
layout(constant_id = 0) const int op_type = 0;
#define shape_constant_id_offset 1
layout(constant_id = shape_constant_id_offset + 0) const int aw = 0;
layout(constant_id = shape_constant_id_offset + 1) const int ah = 0;
layout(constant_id = shape_constant_id_offset + 2) const int ac = 0;
layout(constant_id = shape_constant_id_offset + 3) const int acstep = 0;
layout(constant_id = shape_constant_id_offset + 4) const int bw = 0;
layout(constant_id = shape_constant_id_offset + 5) const int bh = 0;
layout(constant_id = shape_constant_id_offset + 6) const int bc = 0;
layout(constant_id = shape_constant_id_offset + 7) const int bcstep = 0;
layout(constant_id = shape_constant_id_offset + 8) const int outw = 0;
layout(constant_id = shape_constant_id_offset + 9) const int outh = 0;
layout(constant_id = shape_constant_id_offset + 10) const int outc = 0;
layout(constant_id = shape_constant_id_offset + 11) const int outcstep = 0;

layout(binding = 0) readonly buffer a_blob {{ sfp a_blob_data[]; }};
layout(binding = 1) readonly buffer b_blob {{ sfp b_blob_data[]; }};
layout(binding = 2) writeonly buffer top_blob {{ sfp top_blob_data[]; }};

layout(push_constant) uniform parameter {{
    int aw; int ah; int ac; int acstep;
    int bw; int bh; int bc; int bcstep;
    int outw; int outh; int outc; int outcstep;
}} p;

void main() {{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= psc(outw) || gy >= psc(outh) || gz >= psc(outc)) return;

    // broadcast: clamp to each side's actual extent
    int ax = min(gx, psc(aw) - 1);
    int ay = min(gy, psc(ah) - 1);
    int az = min(gz, psc(ac) - 1);
    int bx = min(gx, psc(bw) - 1);
    int by = min(gy, psc(bh) - 1);
    int bz = min(gz, psc(bc) - 1);

    int ai = az * psc(acstep) + ay * psc(aw) + ax;
    int bi = bz * psc(bcstep) + by * psc(bw) + bx;
    int gi = gz * psc(outcstep) + gy * psc(outw) + gx;

    afp v1 = buffer_ld1(a_blob_data, ai);
    afp v2 = buffer_ld1(b_blob_data, bi);
    afp res;
    if (op_type == 0) res = v1 + v2;
    if (op_type == 1) res = v1 - v2;
    if (op_type == 2) res = v1 * v2;
    if (op_type == 3) res = v1 / v2;
    // ... add the ops your base BinaryOp implements
    buffer_st1(top_blob_data, gi, res);
}}
```

Host forward (NOT inplace): populate bindings[0/1/2], set all 12 push constants,
dispatcher = out shape.
"""


_VULKAN_TEMPLATE_REDUCTION = """\
=== PER-OP TEMPLATE: reduction along an axis (Sum / Max / Mean / ...) ===

Two-pass or one-workgroup-per-slice pattern. Simplest correct approach: one
GPU thread per OUTPUT element, loop over the reduced axis inside the shader.

Shader (axis = spec-const, one thread per output element):
```glsl
#version 450
layout(constant_id = 0) const int axis = 0;    // ncnn axis: 0=c, 1=h, 2=w (for dims=3)
layout(constant_id = 1) const int op_type = 0; // 0=sum 1=max 2=min 3=mean ...
#define shape_constant_id_offset 2
layout(constant_id = shape_constant_id_offset + 0) const int inw = 0;
layout(constant_id = shape_constant_id_offset + 1) const int inh = 0;
layout(constant_id = shape_constant_id_offset + 2) const int inc = 0;
layout(constant_id = shape_constant_id_offset + 3) const int incstep = 0;
layout(constant_id = shape_constant_id_offset + 4) const int outw = 0;
layout(constant_id = shape_constant_id_offset + 5) const int outh = 0;
layout(constant_id = shape_constant_id_offset + 6) const int outc = 0;
layout(constant_id = shape_constant_id_offset + 7) const int outcstep = 0;

layout(binding = 0) readonly buffer bottom_blob {{ sfp bottom_blob_data[]; }};
layout(binding = 1) writeonly buffer top_blob {{ sfp top_blob_data[]; }};

layout(push_constant) uniform parameter {{
    int inw; int inh; int inc; int incstep;
    int outw; int outh; int outc; int outcstep;
}} p;

void main() {{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= psc(outw) || gy >= psc(outh) || gz >= psc(outc)) return;

    afp acc = afp(0.0);
    int len = (axis == 0) ? psc(inc) : (axis == 1 ? psc(inh) : psc(inw));
    for (int k = 0; k < len; k++) {{
        int ix = (axis == 2) ? k : gx;
        int iy = (axis == 1) ? k : gy;
        int iz = (axis == 0) ? k : gz;
        int ii = iz * psc(incstep) + iy * psc(inw) + ix;
        afp v = buffer_ld1(bottom_blob_data, ii);
        acc = acc + v;    // or max(acc, v), min(acc, v), ...
    }}
    // for mean: acc = acc / afp(len);
    int gi = gz * psc(outcstep) + gy * psc(outw) + gx;
    buffer_st1(top_blob_data, gi, acc);
}}
```

CRITICAL: ncnn's reduction `axis` is dims-relative, and pnnx writes axes in the
final Mat's rank (see analog dict). Verify axis semantics against the base CPU
kernel's for loops BEFORE writing the shader.
"""


_VULKAN_TEMPLATE_CONV = """\
=== PER-OP TEMPLATE: Convolution (direct, unfused, elempack=1) ===

Convolution is the highest-complexity op. For v1 write ONE direct-conv shader
(no winograd, no im2col, no packing). Correctness first, perf later.

Shader (one thread per output element):
```glsl
#version 450
layout(constant_id = 0) const int kernel_w = 0;
layout(constant_id = 1) const int kernel_h = 0;
layout(constant_id = 2) const int dilation_w = 0;
layout(constant_id = 3) const int dilation_h = 0;
layout(constant_id = 4) const int stride_w = 0;
layout(constant_id = 5) const int stride_h = 0;
layout(constant_id = 6) const int bias_term = 0;
#define shape_constant_id_offset 7
layout(constant_id = shape_constant_id_offset + 0) const int inw = 0;
layout(constant_id = shape_constant_id_offset + 1) const int inh = 0;
layout(constant_id = shape_constant_id_offset + 2) const int inc = 0;
layout(constant_id = shape_constant_id_offset + 3) const int incstep = 0;
layout(constant_id = shape_constant_id_offset + 4) const int outw = 0;
layout(constant_id = shape_constant_id_offset + 5) const int outh = 0;
layout(constant_id = shape_constant_id_offset + 6) const int outc = 0;
layout(constant_id = shape_constant_id_offset + 7) const int outcstep = 0;

layout(binding = 0) readonly buffer bottom_blob {{ sfp bottom_blob_data[]; }};
layout(binding = 1) writeonly buffer top_blob {{ sfp top_blob_data[]; }};
layout(binding = 2) readonly buffer weight_blob {{ sfp weight_data[]; }};
layout(binding = 3) readonly buffer bias_blob {{ sfp bias_data[]; }};

layout(push_constant) uniform parameter {{
    int inw; int inh; int inc; int incstep;
    int outw; int outh; int outc; int outcstep;
}} p;

void main() {{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= psc(outw) || gy >= psc(outh) || gz >= psc(outc)) return;

    afp sum = (bias_term == 1) ? buffer_ld1(bias_data, gz) : afp(0.0);
    // weight layout: [outc][inc][kh][kw] flat
    int w_slice = psc(inc) * kernel_h * kernel_w;
    for (int q = 0; q < psc(inc); q++) {{
        for (int y = 0; y < kernel_h; y++) {{
            for (int x = 0; x < kernel_w; x++) {{
                int iy = gy * stride_h + y * dilation_h;
                int ix = gx * stride_w + x * dilation_w;
                int ii = q * psc(incstep) + iy * psc(inw) + ix;
                int wi = gz * w_slice + q * kernel_h * kernel_w + y * kernel_w + x;
                sum = sum + buffer_ld1(bottom_blob_data, ii) * buffer_ld1(weight_data, wi);
            }}
        }}
    }}
    int gi = gz * psc(outcstep) + gy * psc(outw) + gx;
    buffer_st1(top_blob_data, gi, sum);
}}
```

The base Convolution CPU kernel and reference ncnn `convolution_vulkan.cpp` (in
the examples above) show the padded-input handling and bias binding. For v1 you
may assume pad=0 (the base pipeline injects a Padding layer upstream if needed).
"""


_VULKAN_TEMPLATE_MATMUL = """\
=== PER-OP TEMPLATE: matmul / InnerProduct / Gemm ===

Shader: one thread per output (row, col) — 2D dispatch over (out.w, out.h).
```glsl
#version 450
layout(constant_id = 0) const int bias_term = 0;
#define shape_constant_id_offset 1
layout(constant_id = shape_constant_id_offset + 0) const int M = 0;   // rows of A / rows of out
layout(constant_id = shape_constant_id_offset + 1) const int K = 0;   // cols of A / rows of B
layout(constant_id = shape_constant_id_offset + 2) const int N = 0;   // cols of B / cols of out

layout(binding = 0) readonly buffer a_blob {{ sfp a_data[]; }};
layout(binding = 1) writeonly buffer out_blob {{ sfp out_data[]; }};
layout(binding = 2) readonly buffer w_blob {{ sfp w_data[]; }};
layout(binding = 3) readonly buffer bias_blob {{ sfp bias_data[]; }};

layout(push_constant) uniform parameter {{ int M; int K; int N; }} p;

void main() {{
    int gx = int(gl_GlobalInvocationID.x);   // col in [0, N)
    int gy = int(gl_GlobalInvocationID.y);   // row in [0, M)
    if (gx >= psc(N) || gy >= psc(M)) return;

    afp sum = (bias_term == 1) ? buffer_ld1(bias_data, gx) : afp(0.0);
    for (int k = 0; k < psc(K); k++) {{
        // A[gy][k] * W[k][gx]  — check your base's actual layout for W
        sum = sum + buffer_ld1(a_data, gy * psc(K) + k)
                  * buffer_ld1(w_data, k * psc(N) + gx);
    }}
    buffer_st1(out_data, gy * psc(N) + gx, sum);
}}
```
"""


_VULKAN_TEMPLATE_INDEX = {
    # by analog stem — the pipeline routes on `profile.analog_layer`. Anything
    # not in this table falls back to a generic 1-shader elementwise template.
    "absval": _VULKAN_TEMPLATE_ELEMENTWISE,
    "relu": _VULKAN_TEMPLATE_ELEMENTWISE,
    "sigmoid": _VULKAN_TEMPLATE_ELEMENTWISE,
    "tanh": _VULKAN_TEMPLATE_ELEMENTWISE,
    "elu": _VULKAN_TEMPLATE_ELEMENTWISE,
    "hardsigmoid": _VULKAN_TEMPLATE_ELEMENTWISE,
    "hardswish": _VULKAN_TEMPLATE_ELEMENTWISE,
    "clip": _VULKAN_TEMPLATE_ELEMENTWISE,
    "unaryop": _VULKAN_TEMPLATE_ELEMENTWISE,
    "binaryop": _VULKAN_TEMPLATE_BINARYOP,
    "eltwise": _VULKAN_TEMPLATE_BINARYOP,
    "reduction": _VULKAN_TEMPLATE_REDUCTION,
    "convolution": _VULKAN_TEMPLATE_CONV,
    "convolutiondepthwise": _VULKAN_TEMPLATE_CONV,
    "innerproduct": _VULKAN_TEMPLATE_MATMUL,
    "gemm": _VULKAN_TEMPLATE_MATMUL,
    "matmul": _VULKAN_TEMPLATE_MATMUL,
}


def _vulkan_op_template(profile: "KernelProfile | None") -> str:
    if profile is None:
        return _VULKAN_TEMPLATE_ELEMENTWISE
    stem = (profile.analog_layer or "").strip().lower()
    for suf in ("_vulkan", "_arm"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return _VULKAN_TEMPLATE_INDEX.get(stem, _VULKAN_TEMPLATE_ELEMENTWISE)


# Legacy alias — kept so _background() keeps building. Composes the three
# blocks: host conventions, shader dialect manual, per-op template.
VULKAN_LAYER_BACKGROUND = (VULKAN_HOST_SIDE_BACKGROUND + "\n\n"
                           + VULKAN_SHADER_DIALECT_BACKGROUND
                           + "\n\n{op_template}")


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
        # Functional ops on arm need a separate playbook: most weight-prep arm
        # optimizations don't amortise, but NEON vector inner loop / omp / kernel
        # specialisation do. See ARM_FUNCTIONAL_STRATEGY for the dos and don'ts.
        if profile and profile.is_functional:
            addendum += "\n" + ARM_FUNCTIONAL_STRATEGY.format(arm_class=sub_class)
    else:  # vulkan
        # HOST_SIDE + SHADER_DIALECT are op-agnostic; op_template is picked by
        # analog_layer. Composing them gives the LLM: (1) file/class rules,
        # (2) shader macros/pitfalls manual, (3) a working per-op skeleton.
        op_tpl = _vulkan_op_template(profile)
        addendum = VULKAN_LAYER_BACKGROUND.format(
            base_header=base_header, base_class=base_class, vulkan_class=sub_class,
            vulkan_header=(profile.header if profile else "cand_x_vulkan.h"),
            vulkan_file=(profile.file if profile else "cand_x_vulkan.cpp"),
            op_template=op_tpl)
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
    def _pathify(name: str) -> str:
        # Reconstruct the ncnn source-tree location from the basename so the
        # LLM sees the real include path (matters for vulkan: host and shader
        # live in different subdirs, and the base .cpp is a THIRD location).
        if name.endswith("_vulkan.h") or name.endswith("_vulkan.cpp"):
            return f"ncnn/src/layer/vulkan/{name}"
        if name.endswith(".comp"):
            return f"ncnn/src/layer/vulkan/shader/{name}"
        if name.endswith("_arm.h") or name.endswith("_arm.cpp"):
            return f"ncnn/src/layer/arm/{name}"
        return f"ncnn/src/layer/{name}"
    return "\n\n".join(f"----- {_pathify(k)} -----\n{v}" for k, v in examples.items())


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


def _interface_reference_block(task_name: str, model_code: str | None = None) -> str:
    """Inject the ncnn built-in layer interface for this task, when known.

    Returns an empty string for tasks whose analog ncnn layer can't be guessed,
    or whose layer isn't in the dictionary. The KernelAgent stays free to make
    its own decisions in that case (no regression on novel ops).

    For multi-input analogs (BinaryOp/Eltwise/Concat/MatMul/Gemm) the
    broadcasting primer is also appended so the ANALYZER prompt itself sees
    ncnn's inner-axis-first rules — important because the analyzer is what
    decides one_blob_only=false and rank_coverage.

    `model_code` (when provided) disambiguates Gemm/MatMul/Linear/Conv-family
    task names by scanning for `nn.Linear` / `nn.Conv2d` etc. — matters because
    ncnn `Gemm` (transA/transB matmul) ≠ ncnn `InnerProduct` (nn.Linear) and
    using the wrong dictionary entry makes the LLM fabricate analog_layer.
    """
    # late import keeps this lookup optional: tests / standalone callers that
    # don't bootstrap_paths still work, they just won't get the reference block.
    try:
        from lookup import guess_layer_from_task, render_for_prompt
    except ImportError:
        return ""
    layer = guess_layer_from_task(task_name, model_code=model_code)
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

For these ops the kernel is a MULTI-INPUT LAYER. The weight tensors arrive as
EXTRA bottom_blobs at every forward() call — NOT via load_model. This is
mandatory because pnnx writes an empty .ncnn.bin for functional ops; if your
kernel calls `mb.load(...)` it WILL CRASH with
"ModelBin read flag_struct failed 0" at NetOracle / production time. The
LayerOracle is permissive and won't catch the bug, but the end-to-end check
will. Avoid this by simply NOT calling mb.load for functional ops.

How to declare a functional op in the profile:
  - `weights_from_inputs: [1, 2]`   — input[1] and input[2] are weights
                                       (in that order); input[0] is activation
  - `weight_keys: []`                — leave empty (no state_dict path)
  - `one_blob_only: false`           — REQUIRED (multi-input layer)
  - `support_inplace: false`         — REQUIRED (different shapes, can't be inplace)

Concrete mapping for `F.conv2d(x, w, b)` with introspect inputs `[x, w, b]`:
  - weights_from_inputs: [1, 2]
  - weight_keys:         []
  - one_blob_only:       false
  - load_model:          MUST be empty (`return 0;`)  — DO NOT call mb.load
  - forward signature:   `int forward(const std::vector<Mat>& bottom_blobs,
                                       std::vector<Mat>& top_blobs,
                                       const Option& opt) const`
  - inside forward:      `const Mat& x = bottom_blobs[0];
                          const Mat& w = bottom_blobs[1];
                          const Mat& b = bottom_blobs[2];  // optional`

Why multi-input instead of mb.load:
  - pnnx-emitted `.ncnn.param` for a functional Conv has THREE Input lines and
    one Convolution line wired as `3 1 in0 in1 in2 out0`. The bin file is empty.
  - When NetOracle calls `Net.load_model(empty_bin)`, ncnn's standard layer
    `Convolution::load_model` reads a 4-byte flag header from the bin and dies.
  - Our retarget step renames the layer to `Cand_<Op>` but keeps the 3-input
    wiring. So `Cand_<Op>` must accept those weights as bottom_blobs.

When `state_dict` is non-empty and the model is an nn.Module wrapper, stick
with `weight_keys` + `one_blob_only=true` + non-empty `load_model`. Do not set
`weights_from_inputs` in that case.
"""


def analyzer_prompt(task_name: str, model_code: str, intro: dict | None,
                    force_analog_layer: str | None = None) -> str:
    # Hard constraint from the baseline probe: pnnx's actual ncnn layer choice
    # for this op. When provided, force the analyzer to USE this layer's
    # interface (not a "semantically nearer" guess). Prevents the common
    # nn.Linear-with-2D-input case where the LLM picks InnerProduct but pnnx
    # emits Gemm — same Python op, different ncnn param schemas, e2e fails.
    forced_block = ""
    if force_analog_layer:
        forced_block = (
            f"\n=== HARD CONSTRAINT: analog_layer = `{force_analog_layer}` ===\n"
            f"The pnnx baseline probe converted this PyTorch model and the\n"
            f"resulting .ncnn.param uses ncnn layer type `{force_analog_layer}`.\n"
            f"Your `analog_layer` field MUST be exactly `{force_analog_layer}` (case\n"
            f"sensitive). Your `params` keys MUST follow that layer's ID schema\n"
            f"(see the interface dictionary block below); pnnx populates the\n"
            f".ncnn.param using those same IDs, so any mismatch fails end-to-end.\n"
        )
    lookup_name = force_analog_layer or task_name
    ref = _interface_reference_block(lookup_name, model_code=model_code)
    return f"""Analyze a PyTorch operator and plan a from-scratch ncnn base kernel.

Task: {task_name}
PyTorch model:
```python
{model_code}
```

Model introspection (ground truth):
{_introspect(intro)}
{forced_block}
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
        return (f"Write AT LEAST three files: {profile.header}, {profile.file}, and the "
                f"primary GLSL shader {profile.shader}. If your op needs multiple pipelines "
                f"(e.g. BinaryOp with a broadcast fast-path, Convolution with a bias-free "
                f"fast-path), emit ONE .comp file PER pipeline and load them via "
                f"`compile_candidate_shader_by_name(opt, \"<name>\", spirv)` (the runtime "
                f"helper reads them from CANDIDATE_SHADER_DIR). Name them "
                f"{profile.shader.replace('.comp','')}_<variant>.comp for clarity. "
                f"The class must be `{profile.class_name}`. Do NOT write "
                f"DEFINE_LAYER_CREATOR. Do NOT reference LayerShaderType.")
    if profile.backend == "arm":
        return (f"Write exactly two files: {profile.header} and {profile.file}. The class "
                f"must be `{profile.class_name}` and MUST override forward/forward_inplace. "
                f"Do NOT write DEFINE_LAYER_CREATOR (arm registration is automatic).")
    return (f"Write exactly two files: {profile.header} and {profile.file}. The class must be "
            f"`{profile.class_name}` and end the .cpp with DEFINE_LAYER_CREATOR({profile.class_name}).")


def _functional_routing_note(profile: KernelProfile, intro: dict | None) -> str:
    """If the op's weights come from forward INPUTS (functional style), tell the
    coder exactly which bottom_blob index is the activation and which are weights.

    Important: weights stay as bottom_blobs (NOT as mb.load slots). pnnx writes
    an empty .ncnn.bin for functional ops, so any mb.load call crashes at
    NetOracle / production time.
    """
    wfi = getattr(profile, "weights_from_inputs", None) or []
    if not wfi:
        return ""
    in_shapes = (intro or {}).get("input_shapes") or []
    activation_idx = [i for i in range(len(in_shapes)) if i not in wfi]
    parts = [
        "=== FUNCTIONAL OP — MULTI-INPUT LAYER (this op has weights in inputs) ===",
        f"This op is FUNCTIONAL. one_blob_only is FORCED to False; your forward "
        f"signature MUST be `int forward(const std::vector<Mat>& bottom_blobs, "
        f"std::vector<Mat>& top_blobs, const Option& opt) const`.",
        f"load_model() MUST be empty — `return 0;` only. DO NOT call mb.load. "
        f"The pnnx-emitted .ncnn.bin for this op is EMPTY (0 bytes); any "
        f"mb.load call reads from a 0-byte stream and crashes NetOracle with "
        f"\"ModelBin read flag_struct failed 0\".",
        f"bottom_blobs layout (order matches the .ncnn.param input wiring):",
    ]
    for i in range(len(in_shapes)):
        if i in activation_idx:
            parts.append(f"  bottom_blobs[{i}] = ACTIVATION,  shape={in_shapes[i]}")
        else:
            slot = wfi.index(i)
            label = ("weight", "bias", "running_mean", "running_var")[slot] \
                    if slot < 4 else f"weight#{slot}"
            parts.append(f"  bottom_blobs[{i}] = WEIGHT ({label}), shape={in_shapes[i]}")
    parts.append("Inside forward(): read the activation from bottom_blobs[0] and "
                 "the weights from bottom_blobs[1..N]. Index into the weight Mats "
                 "using PyTorch's layout (e.g. conv weight is [out][in][kh][kw]). "
                 "Allocate top_blobs[0] for the output.")
    parts.append("")
    parts.append("=== PARAM IDs ARE NCNN, NOT ONNX (CRITICAL for end-to-end correctness) ===")
    parts.append(
        "The PyTorch model file's docstring may list hyper-params in ONNX style "
        "(group, pads, kernel_shape, strides, auto_pad, dilations). IGNORE THAT "
        "ORDER inside load_param. Use the EXACT ncnn LAYER PARAM IDs from the "
        "interface dictionary block above. pnnx writes the .ncnn.param using "
        "those same ncnn IDs, so the NetOracle / production end-to-end run will "
        "feed your load_param with that schema. Using ONNX order will pass the "
        "per-op LayerOracle (it derives kernel/in/out from the weight tensor "
        "shape) but FAIL end-to-end (wrong output shape, e.g. (3, 11, 32) vs "
        "(16, 32, 32) because stride/dilation read from wrong IDs).")
    parts.append(
        "For Convolution specifically the ncnn IDs are:\n"
        "    num_output    = pd.get( 0, 0);        // NOT 'group'!\n"
        "    kernel_w      = pd.get( 1, 0);\n"
        "    kernel_h      = pd.get(11, kernel_w);\n"
        "    dilation_w    = pd.get( 2, 1);\n"
        "    dilation_h    = pd.get(12, dilation_w);\n"
        "    stride_w      = pd.get( 3, 1);\n"
        "    stride_h      = pd.get(13, stride_w);\n"
        "    pad_left      = pd.get( 4, 0);\n"
        "    pad_right     = pd.get(15, pad_left);\n"
        "    pad_top       = pd.get(14, pad_left);\n"
        "    pad_bottom    = pd.get(16, pad_top);\n"
        "    bias_term     = pd.get( 5, 0);\n"
        "    weight_data_size = pd.get( 6, 0);\n"
        "    group         = pd.get( 7, 1);        // only for DepthWise\n"
        "    dynamic_weight= pd.get(19, 0);\n"
        "For other layers use the param table at the top of this prompt verbatim.")
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


def _split_vulkan_code_book(code_book: dict[str, str]) -> str:
    """Render code_book with SHADER files in their own top-level section, so the
    LLM sees at a glance which .comp needs the fix vs which .cpp/.h is host-side.
    Non-vulkan callers use the default `----- name -----` layout via the caller.
    """
    hosts = {n: c for n, c in code_book.items() if not n.endswith(".comp")}
    shaders = {n: c for n, c in code_book.items() if n.endswith(".comp")}
    parts = []
    if hosts:
        parts.append("=== HOST (C++) ===")
        parts.extend(f"----- {n} -----\n{c}" for n, c in hosts.items())
    if shaders:
        parts.append("=== SHADER (GLSL) ===")
        parts.extend(f"----- {n} -----\n{c}" for n, c in shaders.items())
    return "\n\n".join(parts) or "(none)"


def _classify_vulkan_feedback(feedback: str) -> str | None:
    """Return the runner's error category if the feedback carries one, else None.

    The vulkan oracle stamps `category=<label>` into the error text (see
    `VulkanLayerOracle._classify_runner_rc`). Labels we care about:
    shader_compile, pipeline_create, gpu_dispatch, no_vulkan_device.
    """
    import re as _re
    m = _re.search(r"category=(shader_compile|pipeline_create|gpu_dispatch|no_vulkan_device|runner_other)",
                   feedback or "")
    return m.group(1) if m else None


def debugger_prompt(phase: str, profile: KernelProfile, code_book: dict[str, str],
                    feedback: str, memory: str, intro: dict | None) -> str:
    framing = _PHASE.get(phase, "The kernel failed.")
    if profile.backend == "vulkan":
        cur = _split_vulkan_code_book(code_book)
    else:
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
    # Route on the vulkan runner's structured error category if the feedback
    # carries one. shader_compile → focus on .comp; pipeline_create → focus on
    # host-side descriptor/spec-const setup; gpu_dispatch → focus on dispatcher
    # shape / bindings / OOB in shader.
    if profile.backend == "vulkan" and phase in ("compile_repair", "numeric_repair"):
        cat = _classify_vulkan_feedback(feedback)
        if cat == "shader_compile":
            extra += ("\nERROR CATEGORY: shader_compile — the .comp file did NOT compile to "
                      "SPIR-V. The glslang errors above cite .comp LINE NUMBERS. Fix the GLSL "
                      "syntax/semantics in the .comp file. DO NOT touch the .h/.cpp unless a "
                      "binding/push-constant declaration there needs to match a shader change."
                      "\n\nIF the error is `'NAME' : undeclared identifier` from a psc() call: "
                      "psc(x) expands to `(x==0 ? p.x : x)` so `x` MUST BE DECLARED AS BOTH "
                      "a spec-const `layout(constant_id = OFFSET+K) const int NAME = 0;` AND a "
                      "push-constant `int NAME;` inside `layout(push_constant) ... }} p;`. "
                      "Missing either is the #1 shader-compile fail. Fix by adding the matching "
                      "spec-const declaration for every psc(x) name.")
        elif cat == "pipeline_create":
            extra += ("\nERROR CATEGORY: pipeline_create — the shader compiled but Vulkan "
                      "rejected the pipeline. Check: (1) specialization vector length matches "
                      "the shader's `layout(constant_id=N)` count; (2) push_constant range in "
                      "the .cpp `vk_constant_type constants[K]` matches shader's push_constant "
                      "struct byte size; (3) `layout(binding=k)` in the shader vs `bindings[k]` "
                      "count in the .cpp; (4) local_size_xyz set (default 3D leaves elements "
                      "unprocessed for 1D dispatchers).")
        elif cat == "gpu_dispatch":
            extra += ("\nERROR CATEGORY: gpu_dispatch — forward() returned nonzero (MoltenVK "
                      "or the GPU driver hit an error at record/submit time). Common causes: "
                      "dispatcher shape doesn't cover all elements (`dispatcher.w = n; h=1; c=1` "
                      "for 1D — but MUST use w=blob.w, h=blob.h, c=blob.c for a 3D shader); "
                      "bindings vector has fewer entries than the shader expects; OOB access "
                      "in the shader crashed the driver (add `if (gi >= psc(n)) return;`).")
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
