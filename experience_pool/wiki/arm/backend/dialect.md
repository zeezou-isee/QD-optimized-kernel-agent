# ARM NEON dialect for ncnn kernels

Reference for kernel authors targeting ncnn ARM layers on Apple Silicon /
ARMv8. Distilled from `ncnn/src/layer/arm/*_arm.cpp` and `arm_usability.h`.

## Vector width & registers

| Symbol | Value on Apple M5 | Meaning |
| --- | --- | --- |
| `VEC_BITS` | 128 | NEON register width (fixed; SVE not enabled in ncnn's ARM path) |
| `FP32_PER_VEC` | 4 | `float32x4_t` |
| `VECTOR_REGS` | 32 | v0..v31; treat >16 as spill-prone in a hot inner loop |
| `CACHE_LINE` | 128 | Apple-specific (most ARM cores use 64) |
| `L1D` / `L2` | 128 KiB / 16 MiB | P-core; E-core has 64 KiB / 6 MiB |

Feature toggles come from the hardware profile (all 1 on M5):
`HAS_DOTPROD` (sdot/udot), `HAS_ASIMDHP` (fp16 arithmetic — `float16x8_t`),
`HAS_BF16`, `HAS_I8MM`.

## Load / store

| Intrinsic | Use |
| --- | --- |
| `vld1q_f32(p)` / `vst1q_f32(p, v)` | contiguous 4×fp32 load/store |
| `vld1q_dup_f32(p)` | broadcast a single fp32 into all 4 lanes (bias) |
| `vld2q_f32` / `vld3q_f32` / `vld4q_f32` | de-interleaved load (2, 3, 4 streams) — matches NC4HW4 packing |
| `vld1q_f16(p)` | contiguous 8×fp16 (needs `HAS_ASIMDHP`) |
| `__builtin_prefetch(p, 0, 3)` | rw=0, temporal=3 — cheap hint before a long-stride stream |

## Arithmetic (fp32)

| Intrinsic | Op |
| --- | --- |
| `vaddq_f32(a,b) vsubq_f32 vmulq_f32 vdivq_f32` | element-wise |
| `vmlaq_f32(acc,a,b)` | acc + a*b (fused; the workhorse of conv/gemm) |
| `vfmaq_f32(acc,a,b)` | true FMA — prefer over `vmlaq` where available |
| `vmaxq_f32 vminq_f32 vabsq_f32 vnegq_f32` | element-wise |
| `vrecpeq_f32` + `vrecpsq_f32` | reciprocal estimate + Newton step |
| `vrsqrteq_f32` + `vrsqrtsq_f32` | rsqrt estimate + step |

## Reductions

| Intrinsic | Op |
| --- | --- |
| `vaddvq_f32(v)` | horizontal sum (returns scalar) |
| `vmaxvq_f32(v) vminvq_f32(v)` | horizontal max/min |
| `vpaddq_f32(a,b)` | pairwise add (interleaved) |

## Permute / lane

| Intrinsic | Use |
| --- | --- |
| `vgetq_lane_f32(v,i)` / `vsetq_lane_f32(x,v,i)` | scalar in/out of lane i |
| `vextq_f32(a,b,n)` | shift concat: lanes n..3 of a || 0..n-1 of b (rotates) |
| `vzipq_f32 vuzpq_f32 vtrnq_f32` | zip/unzip/transpose 2×4→4×2 |
| `vrev64q_f32` | reverse within 64-bit halves |

## Dot product (needs `HAS_DOTPROD`)

| Intrinsic | Op |
| --- | --- |
| `vdotq_s32(acc, a_s8, b_s8)` | 4-way s8 dot into s32 lanes |
| `vdotq_lane_s32(acc, a, b, lane)` | dot vs one lane of b, broadcast |

## fp16 arithmetic (`HAS_ASIMDHP` = 1 on M5)

| Intrinsic | Op |
| --- | --- |
| `vaddq_f16 vmulq_f16 vfmaq_f16` | fp16 add / mul / FMA (8 lanes) |
| `vcvt_f16_f32(v)` / `vcvt_f32_f16(v)` | 4-lane f16↔f32 conversion |

## Common ncnn arm idioms

- **`Mat.cstep` gap**: channel stride in ncnn is `alignSize(w*h*elemsize, 16) /
  elemsize`, NOT `w*h`. Iterate channels via `mat.channel(c).data`; do not
  compute `c*w*h`. Miss this and you overwrite adjacent-channel data on
  non-multiple-of-4 spatial sizes.
- **Scalar tail**: after the 4-wide inner loop, run a scalar epilog for
  `count % 4` elements. Symptom of forgetting: last N elements of the last
  channel wrong (E6 pattern "last-N-scalar-tail").
- **`elempack`**: at OptimizeAgent level assume `elempack = 1` (the harness
  validates that way). packed paths (`_pack4`, `_pack8`) are an advanced
  variant, not the default.
- **Threading**: `#pragma omp parallel for num_threads(opt.num_threads)`.
  Parallelize on the outermost independent axis (usually channels). Skip omp
  when `channels * work_per_channel < ~10k ops` — overhead dominates.

## Guarding for feature-gated code

```cpp
#if __ARM_NEON
    // NEON body
#endif
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // float16x8_t body (matches HAS_ASIMDHP)
#endif
#if __ARM_FEATURE_DOTPROD
    // vdotq_s32 body (matches HAS_DOTPROD)
#endif
```

ncnn injects the right compile flags per translation unit
(`convolution_arm_asimdhp.cpp`, `convolution_arm_asimddp.cpp`) so these guards
are almost always true in the corresponding `.cpp`. Do NOT put fp16/dotprod
bodies in the base `*_arm.cpp` — they will fail to compile on hosts without
those flags.

## Cross-framework conventions (XNNPACK / ACL)

How the vendor CPU stacks structure NEON kernels — adopt the *ideas*.

- **MR×NR register-tile micro-kernels.** XNNPACK names every GEMM/IGEMM
  kernel `*-gemm-MRxNR-*`; the `MR×NR` block of outputs lives in registers,
  accumulated over the reduction axis. The variant is chosen at runtime by
  ISA (`cpuinfo`) + shape. ExecuTorch BLAS confirms the shape: register
  blocked, M-unroll 4, no cache tiling. See `heuristics/tiling_and_packing.md`.
- **ISA tiers.** `cpuinfo` distinguishes NEON → NEON-DOT (dotprod/i8mm) on
  ARM; a specialized kernel is installed per tier at runtime. Guard your
  fp16/dotprod path on `HAS_ASIMDHP` / `HAS_DOTPROD` accordingly.
- **Constant weights → prepack once, free the source** (ACL `prepare()`).
  Do not put the pack in the hot loop. See `heuristics/tiling_and_packing.md`.
- **NHWC is the accelerated layout** in ACL; NCHW needs a one-time weight
  permute. ncnn's default here is unpacked NCHW (`elempack=1`) — layout is a
  first-class BD axis (`bd_axes/memory_bound.md`).

## References

- `ncnn:src/layer/arm/arm_usability.h` — helpers over intrinsic names, keeps
  bodies portable across compilers.
- `ncnn:src/layer/arm/neon_mathfun.h` — `exp_ps / log_ps / sin_ps / tanh_ps`
  vector transcendentals for activations.
- `ncnn:src/layer/arm/convolution_arm.cpp:L1..L60` — canonical dispatch shape:
  choose direct / packed / im2col-gemm / winograd23 / winograd43 by kernel &
  dilation & stride & elempack.
