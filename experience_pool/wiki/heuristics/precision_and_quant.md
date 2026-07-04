# Heuristics — precision & quantization

Cross-framework rules for choosing precision — the single highest-leverage
move in the `mixed` regime because it cuts bytes, moves the roofline ridge,
AND can unlock special instructions at once. Generic, not per-operator.

## fp16 storage and arithmetic are separate toggles

- **Storage fp16** halves bytes moved (memory-bound win regardless of
  compute). **Arithmetic fp16** can double compute throughput but only on
  capable hardware. ncnn splits them: `NCNN_fp16_storage` vs
  `NCNN_fp16_arithmetic`. XNNPACK splits them too:
  `XNN_FLAG_FORCE_FP16_INFERENCE` vs `XNN_FLAG_HINT_FP16_INFERENCE`
  (`LiteRT/tflite/delegates/xnnpack/xnnpack_delegate.cc:1236,1251`).
- **Prior**: in memory-bound regime, fp16 *storage* is a cheap win even if
  you keep fp32 arithmetic. In compute-bound, fp16 *arithmetic* is the
  lever — but only under the availability guard.

## Always accumulate in fp32 (universal rule)

Every framework keeps fp32 accumulators even when inputs/storage/compute
are fp16 or bf16:
- ExecuTorch reduces fp16/bf16 in fp32 (`executorch/kernels/optimized/.../op_sum.cpp`;
  `moments_utils.h`), and its coopmat path accumulates fp32.
- LiteRT's `F32_F16` precision mode is defined as "compute in F16 but
  Convolution / DepthwiseConv / FullyConnected / ConvolutionTransposed
  keep the **accumulator in F32**"
  (`LiteRT/tflite/delegates/gpu/common/precision.h:26-32`).

**Prior**: any reduction, matmul, softmax, LSE, or normalization in fp16
MUST accumulate in fp32 — otherwise expect `E6_NUMERICAL_INSTABILITY`.
Bake `mixed-precision accumulation` into the proposal, not full-fp16.

## fp16 availability tiers

- Native fp16 arith needs **ARMv8.2 FP16** (Pixel 3+, Galaxy S9+, Apple
  A11+, Apple Silicon, Snapdragon 850+). x86 AVX2 only **emulates** fp16
  (compute in fp32, round-trip to fp16 storage)
  (`LiteRT/tflite/delegates/xnnpack/README.md:548-586`).
- On the wiki's hardware profiles this is the `HAS_ASIMDHP` (arm) /
  `HAS_FP16` (vulkan) flag. Guard every fp16-arith proposal on it.

## int8 quantization structure (generic)

When int8 pays (−75% size, ~3× CPU throughput on capable HW), the kernel
structure is fixed across frameworks:

- **int32 accumulator**, then requantize with a **fixed-point
  multiply-shift-clamp**. ExecuTorch/CMSIS model
  (`executorch/backends/cortex_m/passes/passes_utils.py:114-131`):
  ```
  product = (acc << shift) * multiplier + (1 << 30)
  result  = product >> 31              # round-half-away-from-zero
  result  = clamp(result, -128, 127)   # int8
  ```
- Multiplier/shift derived ahead-of-time from the scale:
  `mantissa, shift = frexp(scale); q_fixed = round(mantissa * (1<<31))`
  (`:254-265`), so `scale ≈ multiplier · 2^shift / 2^31`.
- **Symmetric weights** (weight zero-point = 0); asymmetric weights are
  rejected. **Per-channel** quant for conv / depthwise / conv-transpose
  filters, **per-tensor** for everything else; **bias is int32**
  (`executorch/backends/cortex_m/quantizer/quantization_configs.py:28-53`).
- Activations can be lowered to a **256-entry int8 LUT** for cheap
  nonlinearities (`aten_to_cortex_m_pass.py:244-267`).
- Prefer signed **QS8**; unsigned QU8 "may perform suboptimally on NEON
  DOT" (`LiteRT/.../xnnpack/README.md:620-633`). Post-training
  dynamic-range quant is NOT supported by XNNPACK — use static or
  dynamic-per-op.

## Precision tradeoff table (LiteRT, verbatim)

`LiteRT/tflite/g3doc/performance/model_optimization.md:90-146`:

| Scheme | Size | Speed / target |
| --- | --- | --- |
| dynamic-range int8 | −75% | 2–3× CPU |
| full-integer int8 | −75% | 3×+, EdgeTPU + Hexagon |
| float16 | −50% | GPU acceleration |
| int16-act + int8-weight | — | better accuracy but "noticeably slower… lack of optimized kernel", incompatible with HW delegates |

## Decision order

1. Memory-bound & fp16 available → fp16 **storage** first (cheap bytes cut).
2. Compute-bound & fp16 available → fp16 **arithmetic** + fp32 accumulate.
3. Accuracy budget tight → stay fp32, or int16-act/int8-weight (slow).
4. int8 acceptable & static weights → full-integer with the requant
   structure above; prefer QS8; per-channel for conv.
5. Always: fp32 accumulators, guard on `HAS_*` flags.

## Sources

`primitives/reduce_compute.md` (precision primitive); `references/frameworks.md`.
