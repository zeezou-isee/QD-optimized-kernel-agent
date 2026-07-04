# Optimization primitives — reduce computation

Primitives that lower the raw FLOP / MAC count. Effective mainly when the
kernel is **compute-bound** or **mixed**; on memory-bound kernels these move
the ridge point but do not reduce wall time unless combined with a memory
primitive. All primitives are algorithm-agnostic descriptions of a **move**
in the search space, not recipes for any specific operator.

## Primitives

### Algebraic simplification / strength reduction / expression reordering
Rewrite the arithmetic to fewer or cheaper ops without changing the result
type. Examples: distributive law to hoist an invariant, replacing `x*x` with
`fma(x, x, 0)`, replacing `x/y` with `x * (1/y)` when `1/y` is loop-invariant,
scalar folding of constants revealed after specialization. Bit-exact.

### Algorithm substitution
Replace the mathematical algorithm entirely. The substitution changes the
FLOP count but preserves the operator's function. Common substitutions:
- FFT for very long linear convolution (asymptotic `O(n log n)` vs `O(n²)`)
- Winograd for small-tile 2D convolution (fewer multiplies at cost of more adds)
- Strassen / recursive block-decomposition for large matrix multiply
- Low-rank factorization for approximated linear layers
Not bit-exact when the algorithm changes precision behavior (Winograd has
~1-2 bit relative error per pass; FFT of real signals introduces round-off).

### Operator fusion
Merge two or more adjacent kernels so their intermediate result never
materializes to memory. Two flavors:
- **Element-wise fusion**: chain of element-wise ops (add + relu + mul)
  becomes one pass; saves memory bandwidth, not FLOPs.
- **Producer/consumer fusion**: a reduction feeds a broadcast (softmax
  numerator / denominator pattern), or bias+activation folded into a GEMM
  epilog. Saves both bandwidth and kernel launch overhead.

### Precision reduction / quantization
Use narrower types to cut both FLOP and memory cost:
- **fp16 / bf16**: half the storage; hardware may double throughput on
  ARM's `HAS_ASIMDHP`, Vulkan's `HAS_FP16`, cooperative-matrix cores.
- **int8**: quarter the storage; requires quantization scale calibration.
  Enables `HAS_DOTPROD` / `HAS_I8MM` special instructions.
- **mixed-precision accumulation**: keep fp32 accumulators for numerical
  stability while inputs & storage are fp16 (mandatory for softmax, LSE,
  and any deep GEMM chain — otherwise expect `E6_NUMERICAL_INSTABILITY`).

### Sparsity
Skip work on structurally-zero entries. Two regimes:
- **Structured** (2:4, block-sparse, N:M): hardware-friendly, deterministic
  index pattern.
- **Unstructured** (arbitrary zero pattern): needs indirection, only wins
  at extreme sparsity (>90%) on general hardware.

## When to reach for these

- Compute-bound + close to roofline → algorithm substitution is the only lever
  left; algebraic simplification / fusion give small linear gains.
- Compute-bound + far from roofline → prefer parallelism / hardware
  primitives first; algorithm substitution is a bigger commitment.
- Memory-bound → these primitives don't help alone. Fusion is the one
  exception — it reduces bandwidth by killing intermediates.
- Mixed regime → precision reduction is the highest-leverage single move:
  it moves the roofline ridge AND cuts bytes.

## Interactions and pitfalls

- Fusion + tiling fight each other when the fused chain increases per-tile
  register pressure past `VECTOR_REGS`.
- Winograd + int8 needs careful transform-domain quantization; naive
  int8 winograd overflows.
- Precision reduction interacts with numerical stability; softmax /
  logsumexp / normalization pre-max-subtract must be applied at fp32 even
  when the kernel is otherwise fp16.
- Sparsity requires a re-encoding pass; total cost includes the encode. On
  the mobile scale (single-forward inference) the encode usually eats the
  savings unless the sparsity pattern is precomputed offline.

## Cross-framework evidence

- **fp32 accumulation is universal** even under fp16/bf16 compute — LiteRT
  `F32_F16` mode keeps conv/fc accumulators in F32; ExecuTorch reduces
  fp16/bf16 in fp32. See `heuristics/precision_and_quant.md`.
- **Fusion requires a single consumer**: both armnn (fold trailing
  Activation into conv/fc/bn) and ExecuTorch XNNPACK (fuse only
  relu/hardtanh when the GEMM has one user) gate fusion on the producer
  having exactly one downstream use.
- **Winograd is a precision opt-in** (armnn only picks it under
  `fast_math`); never bit-exact. See `heuristics/algorithm_selection.md`.
- int8 requant has a fixed cross-framework structure (int32 accumulator +
  fixed-point multiply-shift-clamp) — see `heuristics/precision_and_quant.md`.

## Source

`AgentDesign/prologue/算子优化-问题建模与体系设计.md:23-31` (降计算量 / 提计算效率);
cross-framework grounding in `heuristics/` (see `references/frameworks.md`).
