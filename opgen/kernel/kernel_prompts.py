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
"""

ARM_LAYER_BACKGROUND = """\
THIS IS AN ARM (NEON) BACKEND KERNEL — it SUBCLASSES the base layer (which is
already written & verified). It is compiled together with the base .cpp and run
with PACKING ON (NC4HW4, elempack=4). Be numerically identical to the base op.

Conventions (CRITICAL):
- Header: `#include "{base_header}"` then `class {arm_class} : public {base_class}`.
  Override ONLY the forward method(s); inherit params/weights/flags from the base
  constructor by calling nothing special — the base ctor already ran.
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


def _background(backend: str, profile: "KernelProfile | None" = None) -> str:
    if backend != "arm":
        return NCNN_LAYER_BACKGROUND
    base_class = (profile.base_class if profile else "") or "BaseLayer"
    arm_class = (profile.class_name if profile else "") or (base_class + "_arm")
    base_header = f"{base_class.lower().replace('cand_', 'cand_')}.h" if profile else "base.h"
    # derive base header from the base class name deterministically (cand_<x>.h)
    if profile and profile.base_class:
        stem = profile.base_class[len("Cand_"):].lower() if profile.base_class.startswith("Cand_") else profile.base_class.lower()
        base_header = f"cand_{stem}.h"
    arm_addendum = ARM_LAYER_BACKGROUND.format(
        base_header=base_header, base_class=base_class, arm_class=arm_class)
    return NCNN_LAYER_BACKGROUND + "\n" + arm_addendum


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
    parts = [f"input shapes (torch, with batch): {intro.get('input_shapes')}"]
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
def analyzer_prompt(task_name: str, model_code: str, intro: dict | None) -> str:
    return f"""Analyze a PyTorch operator and plan a from-scratch ncnn base kernel.

Task: {task_name}
PyTorch model:
```python
{model_code}
```

Model introspection (ground truth):
{_introspect(intro)}

{NCNN_LAYER_BACKGROUND}

Return ONLY a JSON object (no prose):
{{
  "class_name": "Cand_{task_name}",
  "header": "cand_{task_name.lower()}.h",
  "file": "cand_{task_name.lower()}.cpp",
  "one_blob_only": true,
  "support_inplace": false,
  "params": {{}},                      // {{param_id: concrete value for THIS model}}, e.g. {{"0": 1.0}}
  "weight_keys": [],                   // state_dict keys in load_model order (e.g. ["weight","bias"]); [] if weightless
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


def coder_prompt(profile: KernelProfile, examples: dict[str, str], model_code: str, intro: dict | None) -> str:
    return f"""Write the from-scratch ncnn {profile.backend} kernel for this operator.

{_background(profile.backend, profile)}

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

Write exactly two files: {profile.header} and {profile.file}. The class must be
`{profile.class_name}` and end the .cpp with DEFINE_LAYER_CREATOR({profile.class_name}).

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
    return f"""Repair the from-scratch ncnn {profile.backend} kernel.

Situation: {framing}{extra}

{_background(profile.backend, profile)}

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

State the root cause in 1-3 sentences, then return the COMPLETE corrected files.

{OUTPUT_CONTRACT}"""
