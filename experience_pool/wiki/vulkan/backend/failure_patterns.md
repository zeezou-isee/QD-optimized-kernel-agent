# Vulkan failure codes — preempt before the oracle rejects you

Oracle classifies failures via `opgen/layer_oracle/failure_taxonomy.py`.
Vulkan-specific hints appear when `backend == "vulkan"`.

- **`E1_COMPILE`** — SPIR-V compile or C++ pipeline wrapper compile failed.
  Usual roots: mismatched `sfp` vs `float` (buffer typed as `sfp` but stored
  as `float`), `buffer_ld4` on a 1-wide buffer, `#extension` used without
  `#if ncnn_subgroup_*` guard, missing `layout(push_constant) uniform
  parameter { ... } p;` declaration.
- **`E2_RUNTIME_CRASH`** — pipeline barrier violation, out-of-bounds buffer
  write, `barrier()` inside divergent control flow. Vulkan validation layers
  emit descriptive messages — inspect the runner log before guessing.
- **`E3_SHAPE_WRONG_COUNT`** — output binding sized wrong. The wrapper's
  `top_blob.create(...)` sizes the output; a `.comp` writing beyond will
  crash (E2), writing less shows up here.
- **`E4_LAYOUT_PERMUTED`** — dispatch order swapped axes. Common on 3D
  dispatches when `gx/gy/gz` are mapped to `w/h/c` but the wrapper's
  `record_dispatch(gx=..., gy=..., gz=...)` was called with them in a
  different order.
- **`E5_VALUE_AFFINE`** — sign flip: `op == 7` in binaryop is `v2 - v1`
  not `v1 - v2`. Match `binaryop.comp::main` op-type table exactly.
- **`E6_VALUE_NUMERICAL`** — precision mismatch. Common on fp16 pipelines
  (`atol` default is 2e-3 which usually holds; look for larger errors).
  Also: `subgroupAdd` in a partially-filled subgroup — non-participating
  lanes carry undefined values that leak into the reduction.
- **`E6_NUMERICAL_INSTABILITY`** — NaN/Inf. Usually fp16 overflow on
  softmax / exp / large activations. Fix: subtract max before exp,
  clamp inputs, or drop `NCNN_fp16_arithmetic` for the numerically
  sensitive shader.
- **`E8_DISPATCH_COVERAGE`** (vulkan-only) — the oracle sees output
  elements identical to input, meaning the shader didn't touch them.
  Almost always: missing `if (gx >= psc(outw)) return;` guard, workgroup
  shape math yields fewer invocations than output size, or a 1D dispatch
  over a 3D output.
