# ncnn Vulkan idioms

Backend-generic patterns kernel authors must respect regardless of which
operator they're implementing on the Vulkan path. These are ncnn framework
conventions — violate them and the oracle emits `E8_DISPATCH_COVERAGE`,
`E1_COMPILE`, or `E2_RUNTIME_CRASH`. Not specific to any op family.

## Wrapper vs shader

A vulkan candidate is a **triple**: C++ header (`*_vulkan.h`), C++ wrapper
(`*_vulkan.cpp`), and GLSL compute shader (`*.comp`). The wrapper does
pipeline creation and dispatch; the shader is the kernel. Both are needed.

Wrapper responsibilities:
- Compile the shader via `compile_candidate_shader()` (runtime SPIR-V).
- Declare the specialization list (op-type, shape fields, algorithm knob).
- Compute the dispatch shape and call `record_pipeline`.
- Register the layer with `DEFINE_LAYER_CREATOR`.

Shader responsibilities:
- Own the compute kernel body.
- Use ncnn's typed load/store macros (`buffer_ld1/st1`, `buffer_ld4/st4`).
- Read shape fields through `psc(x)` (never bare `p.x`).

## Dispatch coverage guard

The dispatch rounds up workgroup counts to multiples of the workgroup
shape — some invocations spawn past the output extent. Every shader
`main()` must guard:

```glsl
void main() {
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= psc(outw) || gy >= psc(outh) || gz >= psc(outc))
        return;
    ...
}
```

Miss it → the top-blob retains stale bytes at the last tile; oracle emits
`E8_DISPATCH_COVERAGE` (vulkan-only diagnostic).

## Typed load/store

Never index buffers with bare `[]`. Use ncnn macros so precision (fp16 vs
fp32) tracks the pipeline's `NCNN_fp16_*` toggles:

| Call | Purpose |
| --- | --- |
| `buffer_ld1(buf, i)` → `afp` | scalar load with precision cast |
| `buffer_st1(buf, i, v)` | scalar store |
| `buffer_ld4(buf, i)` → `afpvec4` | 4-lane load; `i` in units of 4 elements |
| `buffer_st4(buf, i, v)` | 4-lane store |
| `buffer_ld8` / `buffer_st8` | 8-lane, requires 8-wide packed storage |

Type prefixes:
- `sfp` = storage precision (in buffer)
- `afp` = arithmetic precision (in register)
- `lfp` = local/shared precision

They alias to `float` under fp32 pipelines and `float16_t` under fp16
pipelines. Never write raw `float`.

## Push constants + specialization

Push constants are declared TWICE:
1. As a `layout(push_constant) uniform parameter { ... } p;` block —
   the runtime path.
2. As specialization constants at `layout(constant_id = shape_offset + N)` —
   the compile-time path (driver re-optimizes SPIR-V when the shape is
   known at `Pipeline::create`).

Access via `psc(x)` — the macro picks the spec value when available and
falls back to `p.x` otherwise. Reading `p.x` directly defeats the
specialization optimization pass.

## Workgroup size declaration

```glsl
layout(local_size_x_id = 233, local_size_y_id = 234, local_size_z_id = 235) in;
```

The IDs are specialization slots — the wrapper picks the actual workgroup
shape at pipeline creation. Typical shapes:
- `(SUBGROUP_SIZE, 1, 1)` — 1D linear traversal (elementwise, reduction)
- `(8, 8, 1)` — 2D spatial (conv workgroup)
- `(4, 4, 4)` — 3D block

Constraints:
- `wg_x * wg_y * wg_z <= MAX_WG_INVOCATIONS`
- `shared_bytes <= MAX_SHARED_MEM_BYTES`

## Shared memory + barriers

```glsl
shared lfp sdata[256];   // MUST be a compile-time size (spec const or literal)
...
sdata[tid] = value;
barrier();
memoryBarrierShared();   // ncnn defensive pattern; emit before barrier() in reductions
```

Do NOT `barrier()` inside divergent control flow — different lanes taking
different branches produce undefined behavior.

## Subgroup ops guards

```glsl
#if ncnn_subgroup_arithmetic
    #extension GL_KHR_shader_subgroup_arithmetic : enable
    ...subgroupAdd(v)...
#endif
```

The ncnn preprocessor tokens `ncnn_subgroup_arithmetic`, `ncnn_subgroup_shuffle`,
`ncnn_subgroup_ballot` mirror the device profile flags. Prefer these
guards over raw `#extension` directives — ncnn negotiates them per
pipeline.

## Cstep on the vulkan side

Same NCHW-with-channel-padding convention as CPU:
```glsl
int gi = gz * psc(outcstep) + gy * psc(outw) + gx;
```
Do NOT use `gz * psc(outh) * psc(outw)` — the padding gap will corrupt
adjacent channels (analog of the CPU `Mat.cstep` gap).

## MoltenVK quirks (macOS host)

- MoltenVK translates Vulkan → Metal. Subgroup ops route through Metal
  simdgroup ops; not all Vulkan subgroup features have Metal equivalents.
- `atan(y, x)` requires an explicit `float()` cast (see
  `ncnn:src/layer/vulkan/shader/binaryop.comp` `#if NCNN_moltenvk` branch).
- Buffer atomics are more limited on Metal than desktop Vulkan.
- Cooperative matrix support is absent on Apple GPUs — always check
  `HAS_COOPMAT` before proposing.

## References

- `ncnn:src/layer/vulkan/shader/binaryop.comp` — canonical elementwise
  shader shape.
- `ncnn:src/layer/vulkan/shader/reduction.comp` — canonical reduction
  shape with subgroup / shared-memory paths.
- `opgen/layer_oracle/failure_taxonomy.py` — the `E8_DISPATCH_COVERAGE`
  diagnostic and other vulkan-specific hints.
