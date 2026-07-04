# Optimization primitives — reduce memory traffic

Primitives that lower bytes moved through the memory hierarchy. This is the
main battleground on mobile (the design doc calls it "移动端的主战场"). Effective
whenever the kernel is **memory-bound**, and often on **mixed** kernels too.
All primitives are algorithm-agnostic search-space moves.

## Primitives

### Tiling / blocking
Partition the computation so a working set fits in a specific level of the
memory hierarchy (register file → L1 → L2 / GPU shared memory → global).
Effectiveness comes from **reuse** — an element loaded once serves many
operations before being evicted.

Levels observed:
- **Register-blocking**: innermost loop keeps a small dense block in vector
  registers (`TILE_M × TILE_N` accumulators, `UNROLL_K` for depth). Bounded
  by `VECTOR_REGS`.
- **L1 / shared-mem tiling**: mid-loop block sized so `tile_bytes ≤ L1D`
  (arm) or `≤ MAX_SHARED_MEM_BYTES` (vulkan).
- **L2 / cache-line tiling**: outer tile large enough for locality across
  L1 evictions.

Rule of thumb: two nested levels of tiling capture ~90% of the benefit; the
third level yields little on cache-friendly hardware.

### Data-layout transforms
Reorder how tensor elements live in memory so subsequent reads become
contiguous or vector-friendly. Common transforms:
- **NCHW ↔ NHWC**: matches the dominant traversal axis to the innermost
  memory stride.
- **im2col / im2row**: expand a convolution's sliding window into an
  explicit matrix so the compute becomes a plain GEMM (trades memory for
  a much friendlier kernel shape).
- **Packing** (`NC4HW4`, `NC8HW8`): interleave channels in groups of
  `PACK` so a vector load naturally consumes `PACK` channels at once.
  On ncnn ARM: `elempack ∈ {1, 4, 8}`; on ncnn Vulkan: `sfpvec4`, `sfpvec8`.

Layout transforms are **not free**: the layer boundary between two layouts
costs a full pass. Prefer to stage the entire subgraph in one layout, or
absorb the transform into the load pattern.

### Locality via loop transforms
- **Loop interchange**: swap loop nests so the innermost loop walks the
  contiguous axis. Trivial but often forgotten under heavy templating.
- **Loop skewing**: shift indices across nested loops to expose reuse in
  stencils and recurrences.

### Register / shared-memory reuse
Once a tile is in fast storage, extract as many reads as possible before
eviction. Two patterns:
- **Data reuse across output points** (conv / GEMM): a loaded input block
  contributes to many outputs; keep it resident until all consumers fired.
- **Data reuse across time** (double buffering): while stage N computes,
  prefetch stage N+1 into a second buffer.

### Memory coalescing / alignment
On GPUs: adjacent threads in a warp / subgroup must read adjacent addresses
for a single memory transaction. Broken by stride ≠ 1 or misaligned base.
On CPUs: unaligned NEON loads (`vld1q_f32` on non-16-byte-aligned pointer)
are legal on ARMv8 but cost extra cycles on some cores; align to
`CACHE_LINE` when possible.

### Fusion of intermediates
See `reduce_compute.md::Operator fusion` — element-wise fusion is
primarily a bandwidth optimization (kills a full read+write pass of the
intermediate), listed there for taxonomy but belongs conceptually here.

### Software prefetching
Explicit `__builtin_prefetch(ptr, 0, 3)` (arm) issues a hint N cache lines
ahead of use. Wins on regular strided access with long latencies. Wastes
cycles when the hardware prefetcher already has the pattern.

## When to reach for these

- Memory-bound regime → this is the primary lever. Tiling + layout are
  first-class moves.
- Compute-bound regime → still relevant: bad locality can starve the
  compute units. Register-blocking is universal; broader tiling is a
  secondary concern.
- Mixed → tiling + fusion + precision reduction stack additively.

## Interactions and pitfalls

- Tiling + vectorization: tile dimensions must be multiples of vector
  width, else scalar tails dominate.
- Layout transform in isolation is often a wash — the transform pass
  costs one bandwidth pass of the tensor.
- Packing (`NC4HW4` / `sfpvec4`) forces every op on the path to speak the
  same packing; a single unpacked op breaks the chain.
- Shared memory tiling on GPU has a **bank conflict** hazard: pad the
  inner dim (`shared float tile[TILE][TILE+1]`) when strides are
  power-of-two multiples of the bank count.
- Prefetch distance is device-specific — too short = redundant, too long =
  eviction before use.

## Cross-framework evidence

- **Channel-pack-by-4 is universal on GPU**: ncnn `NC4HW4`, ExecuTorch
  `kChannelsPacked` (`slices=ceil(C/4)`, vec4 texel), LiteRT `PHWC4`
  (`slices=DivideRoundUp(C,4)`, `CL_RGBA`). Pad channel tail to a multiple
  of 4; `C=5` costs the same as `C=8`. See `heuristics/tiling_and_packing.md`.
- **Weight prepack once at setup, then free the source**: armnn
  `prepare()`+`FreeTensorIfUnused`, XNNPACK content-addressed pack cache
  keyed by (model, arch). The pack cost is amortized and must not sit in the
  measured hot loop.
- **Register blocking before cache tiling**: XNNPACK MR×NR micro-kernels;
  ExecuTorch BLAS is register-blocked with NO cache tiling. Add a
  cache-blocking level only when the working set exceeds L1/L2.

## Source

`AgentDesign/prologue/算子优化-问题建模与体系设计.md:33-38` (降访存);
cross-framework grounding in `heuristics/tiling_and_packing.md`.
