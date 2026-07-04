# Optimization primitives — hardware-specialized instructions

Primitives that exploit micro-architectural features beyond generic SIMD.
Effective only when the corresponding hardware flag is set (see
`arm/hardware/*.json::features` and `vulkan/hardware/*.json::features` /
`extensions_present`). Availability is device-scoped — always guard usage
behind `HAS_*` symbols.

## Primitives

### Specialized MAC / dot-product instructions
Single-cycle multi-lane operations that a pure SIMD can't match:
- **ARM `vdotq_s32(acc, a_s8, b_s8)`**: 4 int8 lanes × 4 groups → one s32
  accumulator per group in one instruction. Guarded by `HAS_DOTPROD`.
- **ARM `HAS_I8MM`**: 8×2 int8 matrix multiply-add in one instruction.
- **ARM `HAS_SME2`**: scalable matrix extension — variable-width outer
  products; usable via inline asm or SME intrinsics.
- **Vulkan cooperative matrix** (`VK_KHR_cooperative_matrix`): warp-wide
  MxNxK matrix fragment ops mapped to Tensor Cores / equivalent. Guarded
  by `HAS_COOPMAT`.
- **x86 dp4a / VNNI**: analogous int8 dot-product; not in scope for the
  arm/vulkan backends here.

### Subgroup / warp-level primitives (GPU)
One-cycle cross-lane communication within a subgroup:
- **`subgroupAdd/Min/Max/Mul(v)`**: subgroup-wide reduction in log2 steps
  without shared-memory round-trip. Guarded by `HAS_SUBGROUP_ARITHMETIC`.
- **`subgroupBroadcast(v, lane)`**: send one lane's value to all.
  Guarded by `HAS_SUBGROUP_BALLOT`.
- **`subgroupShuffle(v, srcLane)` / `subgroupShuffleXor`**: permute lanes.
  Guarded by `HAS_SUBGROUP_SHUFFLE`.
- **`subgroupBallot(cond)`**: bitmask of predicate results across lanes.

Use these to replace shared-memory reduction trees (fewer barriers, fewer
instructions) when the reduction fits in a subgroup.

### Bank-conflict elimination (GPU shared memory)
Shared memory is banked (typically 32 banks of 4 bytes on GPUs). Two lanes
of the same subgroup accessing the same bank on different addresses
serialize. Fix by:
- Padding the inner dim: `shared float tile[N][BANKS+1]`.
- Rearranging the access pattern so lanes always hit distinct banks.
- Using `swizzled` addressing patterns (XOR-based) when the natural stride
  aliases banks.

### Register allocation hints
Compilers usually do a good job, but hot inner loops can benefit from:
- **Reordering to reduce live ranges**: consume a loaded value before
  loading the next.
- **Manual scalar-replacement of array elements**: `float a0 = arr[0]; …`
  tells the compiler these values fit in registers.
- **`__restrict__` on pointer arguments**: allows the compiler to reorder
  loads/stores that would otherwise alias.

### Instruction scheduling
On in-order cores (some E-cores, embedded targets), the source order of
instructions matters. Interleave independent chains to feed each pipeline
port every cycle. Modern out-of-order cores (Apple P-core, most Vulkan
GPUs) reorder for you — this primitive matters less there.

## When to reach for these

- Compute-bound + close to roofline: `dotprod` / cooperative_matrix /
  subgroup ops are the last real levers.
- After exhausting tiling / vectorization / unroll — hardware-specialized
  is what remains.
- **Always check availability first**: guarded behind a `HAS_*` symbol in
  the hardware profile. Missing guard → `E1_COMPILE` or silent SIGILL.

## Interactions and pitfalls

- Specialized instructions produce specific-precision results. `vdotq_s32`
  yields s32; if you accumulate more than 2^31 / 127² ≈ 130k mac-ops per
  lane you overflow.
- Cooperative-matrix fragment shapes are hardware-specific (typically
  8×8×8, 16×16×16). Mismatched shape → `E1_COMPILE`.
- Subgroup ops on partially-filled subgroups: lanes past the workgroup
  bound carry undefined values. Initialize with the identity element
  (0 for `subgroupAdd`, `-FLT_MAX` for `subgroupMax`) before the op, or
  gate on `gl_SubgroupInvocationID < live_count`.
- MoltenVK note: subgroup ops map to Metal simdgroup ops; not all Vulkan
  subgroup features have Metal equivalents. Check the runtime profile,
  don't assume desktop-Vulkan semantics.

## Cross-framework evidence

- **ISA/vendor-tiered runtime dispatch** is universal: XNNPACK selects the
  micro-kernel variant via `cpuinfo` tiers (NEON → NEON-DOT on ARM;
  SSE → AVX → AVX2+FMA → AVX512 on x86); LiteRT GPU branches on
  Adreno/Mali/PowerVR/Apple/NVIDIA; armnn gates SVE/SME on runtime
  `CPUInfo`. Always guard specialized paths behind a `HAS_*` flag.
- **Cooperative-matrix has hard shape gates** (ExecuTorch, verbatim):
  `M%64==0 && N%64==0 && K%32==0`, subgroup==64, buffer storage, and
  `!is_integrated_gpu()`. See `heuristics/algorithm_selection.md`.
- **Signed int8 (QS8) preferred over unsigned (QU8)** on NEON-DOT hardware.

## Source

`AgentDesign/prologue/算子优化-问题建模与体系设计.md:44` (硬件指令 / 微架构层);
`算子优化-问题建模与体系设计.md:400-407` (device query / microbench schema);
cross-framework grounding in `heuristics/` (see `references/frameworks.md`).
