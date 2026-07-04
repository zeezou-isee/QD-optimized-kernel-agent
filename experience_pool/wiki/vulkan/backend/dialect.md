# ncnn shader dialect (GLSL compute for Vulkan)

Reference for authoring `.comp` shaders and their C++ pipeline wrappers under
ncnn's Vulkan backend. Distilled from `ncnn/src/layer/vulkan/shader/*.comp`
and `ncnn/src/layer/vulkan/*_vulkan.cpp`.

## Type-length shorthand

ncnn's shader compiler injects precision-parametric type macros so the same
shader compiles with fp16 or fp32 depending on the pipeline. Prefer these
over raw `float`.

| Macro | fp32 alias | fp16 alias | Use for |
| --- | --- | --- | --- |
| `sfp` | `float` | `float16_t` | **storage** type (in buffers) |
| `afp` | `float` | `float16_t` | **arithmetic** register type |
| `lfp` | `float` | `float16_t` | shared / local memory type |
| `sfpvec4` | `vec4` | `f16vec4` | 4-wide storage |
| `sfpvec8` | `vec4[2]` | `f16vec4[2]` or `f16vec8` | 8-wide storage |
| `afpvec4` / `afpvec8` | `vec4` / `vec8` | `f16vec4` / `f16vec8` | 4/8-wide arith |

Toggled by preprocessor `NCNN_fp16_storage` (buffer type) and
`NCNN_fp16_arithmetic` (register type) which ncnn sets based on
`opt.use_fp16_storage` / `opt.use_fp16_arithmetic` per pipeline.

## Buffer load/store

Never write `buf[i]` directly — use ncnn's typed loads:

| Function | Behavior |
| --- | --- |
| `buffer_ld1(buf, i)` → `afp` | scalar load, correct precision cast |
| `buffer_st1(buf, i, v)` | scalar store |
| `buffer_ld4(buf, i)` → `afpvec4` | 4-lane load; `i` is in **units of 4 elements** |
| `buffer_st4(buf, i, v)` | 4-lane store |
| `buffer_ld8(buf, i)` → `afpvec8` | 8-lane load; requires storage buffer packed as 8-wide |
| `buffer_st8(buf, i, v)` | 8-lane store |
| `image3d_ld1(img, uvw)` / `image3d_ld4(...)` | image variants when the pipeline uses images not buffers |

## Push constant access — `psc()`

Push constants are declared twice — once as `layout(push_constant)` block, once
as specialization constants (`layout(constant_id=…)`). Read them via `psc(x)`
so ncnn can substitute the specialization value when the shape is known at
pipeline creation:

```glsl
if (gx >= psc(outw)) return;      // psc(outw) resolves to spec const if set,
                                  // otherwise falls through to p.outw
```

Do NOT read `p.outw` directly — you defeat the specialization pass and every
dispatch pays the branch cost.

## Specialization constants

Declared as `layout(constant_id = N) const int X = 0;`. Fixed at
`Pipeline::create`, driver re-optimizes SPIR-V. Cheap knob for:

- op type / algorithm variant (`op_type`, `with_scalar`)
- shape fields (`aw`, `ah`, `ac`, `acstep` and out equivalents) via
  `shape_constant_id_offset + N`
- workgroup dimensions when the shape is known

## Workgroup layout

Ncnn shaders declare `layout(local_size_x_id=…, local_size_y_id=…,
local_size_z_id=…)` and let the pipeline decide per-op. Common shapes:

| Shape | Use |
| --- | --- |
| `(SUBGROUP_SIZE, 1, 1)` | 1D linear traversal — elementwise, reduction final pass |
| `(8, 8, 1)` | 2D spatial — most conv variants |
| `(4, 4, 4)` | 3D block — 3D conv, packed conv over `(w, h, c)` |

Constraints:

- `local_size_x * local_size_y * local_size_z <= MAX_WG_INVOCATIONS`
  (1024 on Apple M5)
- workgroup shared memory <= `MAX_SHARED_MEM_BYTES` (32768 on M5)
- push constants block <= `MAX_PUSH_CONSTANTS_BYTES` (4096 on M5)

## Subgroup operations

Enabled via `#extension GL_KHR_shader_subgroup_arithmetic : enable` (or
`_shuffle`, `_ballot`). Check `HAS_SUBGROUP_ARITHMETIC` before using.

| Op | Requires | Use |
| --- | --- | --- |
| `subgroupAdd(v)`, `subgroupMax(v)`, `subgroupMin(v)`, `subgroupMul(v)` | `HAS_SUBGROUP_ARITHMETIC` | 1-round subgroup-wide reduction |
| `subgroupBroadcast(v, lane)` | `HAS_SUBGROUP_BALLOT` | send one lane to all |
| `subgroupShuffle(v, srcLane)` | `HAS_SUBGROUP_SHUFFLE` | permute within subgroup |
| `subgroupBarrier()` | basic | full sub-group sync |

Apple M5 supports all four families (see `apple_m5.json::subgroup`).

## Shared memory & barriers

```glsl
shared lfp sdata[256];         // MUST size at compile time
// ...
sdata[gl_LocalInvocationID.x] = ...;
barrier();                     // workgroup-wide barrier
memoryBarrierShared();         // memory model: shared-memory writes visible
```

- Only `shared` variables are workgroup-shared.
- `barrier()` also implies memory barriers for `shared` writes since Vulkan
  1.1 / GLSL 450 — but ncnn shaders defensively emit `memoryBarrierShared()`
  before the `barrier()` in reduction paths, follow that pattern.
- Bank conflicts: layout `shared lfp sdata[N][BLOCK+1]` (pad the inner dim) when
  the shape would otherwise stride a common power of two.

## Guards & backend quirks

- `#if NCNN_moltenvk` — Metal-translated backend on macOS. Some GLSL builtins
  differ (`atan(y,x)` needs to be routed through `float()` casts). Do NOT
  assume MoltenVK ≡ desktop Vulkan.
- `#if ncnn_subgroup_arithmetic` / `#if ncnn_subgroup_shuffle` — ncnn
  preprocessor tokens exposing the profile; matches the JSON `subgroup`
  entries. Prefer these over `#extension` because ncnn negotiates them
  per-pipeline.
- `#if NCNN_fp16_storage` / `#if NCNN_fp16_arithmetic` — pipeline-time toggles.
  Guard fp16-specific code (e.g. subgroup reductions require
  `GL_EXT_shader_subgroup_extended_types_float16`).

## Common ncnn vulkan idioms

- **`cstep` gap**: `int gi = gz * psc(outcstep) + gy * psc(outw) + gx;` —
  same NCHW-with-channel-padding as CPU side; do not compute
  `gz * psc(outh) * psc(outw)`.
- **Dispatch coverage**: after workgroup indexing, check
  `if (gx >= psc(outw) || gy >= psc(outh) || gz >= psc(outc)) return;`
  before writing. Miss it and the top-blob has unmodified stale bytes at
  the last rounding-up tail (oracle emits `E8_DISPATCH_COVERAGE`).
- **Wrapper vs shader**: your `*_vulkan.cpp` creates the pipeline, decides
  the shader variant (specialization constants), and calls `record_dispatch`.
  Your `.comp` is the kernel. Both are needed; the wrapper is where you set
  `pipeline->create(shader_type, opt, specializations)`.

## Cross-framework conventions (executorch / LiteRT)

How other mature Vulkan stacks structure compute shaders — adopt the
*ideas*, keep ncnn's macros.

- **Variants are code-generated, not hand-written.** ExecuTorch pairs each
  `foo.glsl` template with a `foo.yaml` matrix (DTYPE × PACKING × STORAGE)
  and expands the cartesian product to SPIR-V
  (`executorch/backends/vulkan/.../gen_vulkan_spv.py:830`). ncnn hand-writes
  `_pack4`/`_fp16s` — same mental model (one kernel, many instantiations).
- **Three parametrization channels by change-frequency**: compile-time
  macros (structure) → spec constants (layout/broadcast flags; the
  workgroup `local_size` is itself a spec-const) → push constants (tiny
  per-dispatch scalars); larger metadata via UBO. Matches ncnn's
  spec-const + `psc()` split.
- **Default storage is 3D texture**, buffer opt-in (ExecuTorch
  `suggested_storage_type()=kTexture3D`; LiteRT prefers textures on mobile,
  buffers on desktop — see `idioms.md`).
- **Broadcast = per-input index clamp**, not a separate kernel:
  `in_idx = min(out_idx, in_sizes-1)` (ExecuTorch `binary_op_texture.glsl:77`).
- **Variant name = base + suffixes** (`_texture3d`/`_buffer`,
  `_float`/`_half`/`_int8`) — string assembly, not a table.

## References

- `ncnn:src/layer/vulkan/shader/binaryop.comp` — canonical elementwise shader:
  spec-const op_type, `psc()` for shape, `buffer_ld1/st1`, 3D dispatch,
  coverage guard.
- `ncnn:src/layer/vulkan/shader/reduction.comp` — canonical reduction: shared
  memory 2-pass with subgroup arithmetic fast path.
- `executorch/backends/vulkan/runtime/graph/ops/glsl/` — templated GLSL
  library. `LiteRT/tflite/delegates/gpu/` — GPU delegate.
- `ncnn:src/layer/vulkan/shader/convolution_1x1s1d1_cm.comp` — cooperative
  matrix path (only when `HAS_COOPMAT`; ignore on Apple M5).
- `ncnn:src/layer/vulkan/shader/convolution_packed.comp` — packed conv
  workhorse.
- `ncnn:src/layer/vulkan/binaryop_vulkan.cpp` — pipeline wrapper:
  specialization list, dispatch shape.
