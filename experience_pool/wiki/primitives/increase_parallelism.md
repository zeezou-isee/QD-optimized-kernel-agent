# Optimization primitives — increase parallelism / hide latency

Primitives that expose more independent work to the hardware or overlap
compute with memory access. Effective in both regimes: compute-bound
benefits from more parallel FMAs; memory-bound benefits from more in-flight
loads hiding latency. All primitives are algorithm-agnostic.

## Primitives

### Loop transforms
- **Unroll**: replicate the loop body `UNROLL` times so more independent
  ops sit in the instruction stream. Bounded by `VECTOR_REGS` (accumulator
  spill = defeats the point) and code-size (I-cache pressure).
- **Loop fusion** (jam): merge two loops with the same trip count into
  one, letting register-resident data flow between them.
- **Loop fission**: split one loop into two — sometimes needed to isolate
  a hot inner region from unrelated cold work.
- **Loop reorder**: independent of interchange, sometimes reorders
  iterations to expose reuse or vectorization opportunity.

### Vectorization
Use SIMD instructions to process `VEC` elements per instruction:
- ARM NEON: `float32x4_t` (4 fp32), `float16x8_t` (8 fp16 when
  `HAS_ASIMDHP`), `int8x16_t` (16 int8).
- Vulkan GLSL: `vec4` / `f16vec4` / `sfpvec8` types; auto-lowered by the
  driver to hardware SIMD lanes.
- Sub-word: dotprod (`vdotq_s32`) packs 4 int8 into a single 32-bit
  instruction lane.

Vectorization requires **contiguous access on the innermost axis** and
**tile dimensions divisible by vector width**. Broken by strided access,
non-aligned base pointers (on hostile hardware), or a scalar tail loop.

### Multi-level parallel mapping
Map the iteration space to the hardware parallel hierarchy:
- **CPU**: `#pragma omp parallel for` on the outermost independent axis
  (usually channels or batch). One thread per physical core; false sharing
  destroys speedup below the cache-line grain.
- **GPU dispatch**: grid → workgroup → subgroup → lane. Each level
  distributes work; workgroup-shared memory enables cross-lane reuse.
- **Tensorize**: lower an inner block onto a matrix-multiply hardware
  primitive (Vulkan `VK_KHR_cooperative_matrix`, ARM `HAS_SME2`, x86
  Tensor Cores).

### Pipelining / double buffering
Overlap two stages that would otherwise stall each other:
- **Load / compute overlap**: stage N+1 is loaded into buffer B while
  stage N computes on buffer A. Requires 2× register or shared-memory
  budget for the working set.
- **Software pipeline**: unroll a loop by K iterations and interleave the
  bodies so long-latency ops (loads, FMAs on old hardware) are dispatched
  ahead of the consuming op.

### Split-K / load balancing
When the reduction axis (K in GEMM) dominates and there aren't enough
`(M, N)` output tiles to fill the machine, split the reduction across
multiple threads/workgroups and reduce their partial results at the end.
Trades an extra reduction pass for parallel utilization.

### Instruction-level parallelism (ILP)
Increase independent instruction chains in the inner loop so an
out-of-order or wide-issue core can dispatch multiple per cycle. Techniques:
- Multiple accumulators (`acc0, acc1, acc2, acc3`) instead of one — hides
  FMA latency (typically 3–4 cycles on ARM P-cores).
- Avoid dependency chains where `acc = acc + x` for all iterations serialize.

## When to reach for these

- Any regime — parallelism/latency-hiding is universally applicable, but
  the payoff level depends on where the current bottleneck sits.
- Compute-bound → unroll + multiple accumulators + tensorize hit
  saturating throughput. This is the "final 20%" pass.
- Memory-bound → prefer prefetching + double buffering to expose more
  in-flight loads; more SIMD lanes don't help if bandwidth is saturated.
- Setup bounded (small tensors) → multi-thread overhead can exceed the
  work; check with `OMP_MIN_ELEMS` gates before enabling omp.

## Interactions and pitfalls

- Unroll × register pressure: `UNROLL * (accumulators + loaded values) >
  VECTOR_REGS` → spill → defeats the unroll.
- Vectorization × tail: `size % VEC != 0` forces a scalar epilog; forget
  the epilog and get `E6_VALUE_NUMERICAL` "last-N-scalar-tail" from the
  oracle.
- Multi-thread × small tensor: launch/join cost > useful work; measure.
- Double buffering × shared memory: 2× the tile → may push over
  `MAX_SHARED_MEM_BYTES`; check the constraint before proposing.
- Tensorize × supported flag: cooperative_matrix / SME2 require
  `HAS_COOPMAT == 1` / `HAS_SME2 == 1` in the hardware profile — silently
  broken otherwise.
- Split-K × correctness: partial-sum reduction must be numerically stable
  (Kahan or two-pass) when accumulating fp16.

## Cross-framework evidence

- **CPU is single-threaded by default**; a pool is created only when
  `num_threads > 1` (XNNPACK), work split along one window axis (ACL).
  Parallelize the outermost independent axis; skip for small tensors.
- **GPU workgroup shape has ready-made vendor seed tables** (LiteRT GL/Mali,
  ExecuTorch pickers) keyed by channel-slice count — see
  `heuristics/parallelism_and_workgroup.md`. Use as `local_size` seeds.
- **Multiple fp32 accumulators + unroll-4** is the canonical ILP pattern
  (ExecuTorch BLAS `kF32RegisterPairsPerIteration=4`, 8 registers/iter).
- **Cooperative-matrix is desktop-only** — gated off on integrated/mobile
  GPUs; don't propose it when `HAS_COOPMAT == 0`.

## Source

`AgentDesign/prologue/算子优化-问题建模与体系设计.md:40-44` (提并行 / 隐藏延迟);
cross-framework grounding in `heuristics/parallelism_and_workgroup.md`.
