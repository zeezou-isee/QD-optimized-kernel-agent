# ncnn ARM idioms

Backend-generic patterns kernel authors must respect regardless of which
operator they're implementing. These are ncnn framework conventions on the
ARM CPU path — violate them and the oracle emits `E6_VALUE_NUMERICAL` or
`E2_RUNTIME_CRASH`. Not specific to any op family.

## Mat access patterns

**`Mat.cstep` gap**: channel stride is `alignSize(w*h*elemsize, 16) / elemsize`,
NOT `w*h`. The last row of each channel is padded up to 16-byte alignment;
addressing across channels via `c*w*h` overwrites the padding into the next
channel's first row.

Correct pattern:
```cpp
for (int c = 0; c < channels; ++c) {
    const float* pc = mat.channel(c);        // gap-safe base
    for (int i = 0; i < mat.w * mat.h; ++i)  // walk INSIDE one channel only
        ...pc[i]...;
}
```

Wrong:
```cpp
float* p = mat.data;
for (int i = 0; i < mat.w * mat.h * mat.c; ++i)  // corrupts on non-aligned w*h
    ...p[i]...;
```

Oracle diagnosis: `E6_VALUE_NUMERICAL` with tag `SUSPICION (channel gap):
ncnn pads mat.cstep per channel`.

## Vector loop + scalar tail

Every SIMD inner loop must have a scalar epilog for the `size % VEC` residue:

```cpp
int i = 0;
for (; i + 4 <= n; i += 4) {              // 4-wide fp32 body
    float32x4_t v = vld1q_f32(p + i);
    ...
    vst1q_f32(q + i, out);
}
for (; i < n; ++i)                        // scalar tail — MUST include
    q[i] = ...scalar_op(p[i])...;
```

Oracle diagnosis: `E6_VALUE_NUMERICAL` with tag `NEON pattern: all errors
are in the last N (scalar-tail) elements`.

## Weight load type flag

`mb.load(shape, load_flag)`:
- `flag = 0` — tagged (typed) serialization; ncnn writes a small type-tag
  header, `load` reads and validates it. Used for typed weight arrays.
- `flag = 1` — raw untyped bytes. Used for scalar params, indices, etc.

Mismatch produces sign-flipped, all-wrong values → oracle `E6_VALUE_NUMERICAL`
with tag `WEIGHT-MISALIGNMENT SUSPECT`. Check the interface dictionary
(`opgen/ncnn_interface/experience_pool/backend_ncnn/layer_interfaces.md`)
for the correct flag per weight.

## Threading

```cpp
#pragma omp parallel for num_threads(opt.num_threads)
for (int q = 0; q < channels; ++q) {
    ...per-channel work...
}
```

Rules:
- Parallelize on the outermost independent axis (usually channels).
- Skip omp when `total_work < ~10k ops` — thread launch/join dominates.
- Never parallelize across the reduction axis — races or requires
  per-thread accumulators.
- Guarded by `opt.num_threads` from ncnn's `Option` (do not read env
  vars, do not use `omp_get_max_threads` directly).

## Feature-gated code placement

ncnn compiles fp16 / dotprod / bf16 bodies in **separate translation
units**:
- Base body → `<layer>_arm.cpp` (fp32 only, no fp16 intrinsics).
- fp16 body → `<layer>_arm_asimdhp.cpp` (compiled with `-march=armv8.2-a+fp16`).
- dotprod body → `<layer>_arm_asimddp.cpp` (compiled with `+dotprod`).
- bf16 body → `<layer>_arm_bf16.cpp`.

The base `.cpp` MUST NOT reference `float16x8_t`, `vdotq_s32`, etc. even
under `#if` guards — the compile flags aren't set for the base TU, and
`E1_COMPILE` will fire.

## Elempack assumption

At OptimizeAgent scope: `elempack = 1` (unpacked). The `elempack == 4` /
`elempack == 8` packed paths (`NC4HW4` / `NC8HW8`) exist as separate
opt-in layers with their own conventions — do NOT assume packed layout
unless the kernel explicitly declares it. Attempting NC4HW4 access
patterns on an unpacked Mat gets `E4_LAYOUT_PERMUTED`.

## `#pragma unroll` availability

Both GCC and Clang accept `#pragma unroll N` inside function bodies on
ARM. Some hostile compilers require `_Pragma("unroll(N)")` inside a
`#define` — ncnn's convention is to just use `#pragma unroll` directly.

## References

- `ncnn:src/mat.h` — `Mat` layout, `cstep` definition, `channel()` accessor.
- `ncnn:src/layer/arm/arm_usability.h` — intrinsic wrappers.
- `opgen/layer_oracle/failure_taxonomy.py` — the diagnostic-tag vocabulary
  used above.
