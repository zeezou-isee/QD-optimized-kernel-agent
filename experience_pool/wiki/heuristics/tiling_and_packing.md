# Heuristics — tiling & packing (where to look first)

Cross-framework dispatch/tuning heuristics distilled from ncnn, ARM
Compute Library (via armnn), ExecuTorch, and XNNPACK / LiteRT. These are
**search priors** — starting points and orderings for the optimization
search, not per-operator recipes. When three independent frameworks agree,
treat it as a strong prior; where they diverge, that divergence is itself
a signal that the choice is device- or shape-dependent (worth a search
axis).

## Register blocking comes first, cache tiling second

- Mobile CPU kernels are built as **MR×NR register-tile micro-kernels**:
  the innermost block of `MR` output rows × `NR` output cols lives entirely
  in vector registers, accumulated over the reduction axis. XNNPACK's whole
  micro-kernel library is named this way (`*-gemm-MRxNR-*`); the variant is
  chosen at runtime by ISA + shape.
- ExecuTorch's own BLAS confirms the shape: register-blocked with
  **M-unroll of 4**, **8 vector registers per iteration**, and
  **NO cache-level tiling** — register blocking alone
  (`executorch/kernels/optimized/blas/BlasKernel.cpp:33-37`).
- **Prior**: on CPU, exhaust register blocking (`UNROLL`, `TILE_M`,
  `TILE_N` bounded by `VECTOR_REGS`) before adding a cache-blocking level.
  Add L1/L2 tiling only when the working set clearly exceeds cache — e.g.
  ExecuTorch cache-blocks softmax at `64*1024` bytes, a constant it notes
  was "halved to fit mobile caches"
  (`executorch/kernels/optimized/.../op_log_softmax.cpp:73-77`).

## GPU: pack channels in groups of 4 (universal)

All three GPU stacks store activations as **vec4 texels along the channel
dim**, channel count padded up to a multiple of 4:

- ncnn: `NC4HW4` / `sfpvec4`
- ExecuTorch: `kChannelsPacked`, `slices = DivideRoundUp(C, 4)`, one vec4
  texel per 4 channels (`executorch/backends/vulkan/.../ComputeGraph.cpp:266-280`,
  `indexing_utils.h:165-171`)
- LiteRT: `PHWC4` / `DHWC4`, `slices = DivideRoundUp(C, 4)`, images are
  `CL_RGBA` (`LiteRT/tflite/delegates/gpu/common/task/tensor_desc.cc:177`,
  `api.h:62-75`)

Consequences to bake into proposals:
- A tensor with `C=5` costs the same as `C=8` (both use 2 slices) — LiteRT
  explicitly advises preferring channel counts that are multiples of 4
  (`LiteRT/tflite/delegates/gpu/README.md:155-157`).
- Address the packed dim in texel units, never element units.
- A single unpacked op on the path breaks the whole packed chain — keep
  the subgraph in one packing.

## Weight prepacking: once, at setup, then free the source

Constant weights are reshaped into the kernel's internal (tiled) layout
one time, cached, and the original buffer is released:

- armnn/ACL: `prepare()` packs on first run, then `FreeTensorIfUnused`
  drops the un-packed copy (`armnn/src/backends/neon/workloads/NeonConvolution2dWorkload.cpp:154-166`).
- XNNPACK: pack once during graph setup; content-addressed cache keyed by
  `PackIdentifier{algorithm, weights, bias}`, file-backed mmap, **cache is
  bound to a (model, architecture) pair — never shared across either**
  (`LiteRT/tflite/delegates/xnnpack/README.md:147-190`).
- LiteRT GPU weight layout bundles output slices for register tiling:
  `OGroup` groups multiple O-slices, I and O both padded to 4 — the GPU
  analogue of MR×NR (`LiteRT/tflite/delegates/gpu/common/task/weights_layout.h:26-36`).

**Prior**: if weights are constant, propose a prepack step separate from
the hot loop; the packing cost is amortized and must not appear in the
measured kernel. Constant-vs-dynamic operand-ness selects a different
kernel (constant → prepack path; dynamic → on-the-fly)
(`armnn/.../NeonBatchMatMulWorkload.cpp:59-61`).

## Search-then-memoize is the industry pattern (and this project's core)

Both vendor stacks pick tile/workgroup sizes by **empirical search cached
to disk**, exactly the QD/experience-pool idea:

- armnn CLTuner has 4 levels `None / Rapid / Normal / Exhaustive`, persists
  winning configs to a file and replays them; plus an ML-learned GEMM
  heuristic table (MLGO) keyed on shape
  (`armnn/src/backends/aclCommon/ArmComputeTuningUtils.hpp:18-67`,
  `armnn/src/backends/cl/ClBackendContext.cpp:150-171`).
- LiteRT sweeps candidate workgroups, times each with GPU events, caches
  the winner by kernel fingerprint; also caches compiled OpenCL binaries
  (`LiteRT/tflite/delegates/gpu/cl/cl_operation.cc:195-223`,
  `program_cache.cc:34-55`).

**Prior**: the tile / workgroup / unroll knobs are exactly the ones worth
an inner search; seed them from the tables in
`heuristics/parallelism_and_workgroup.md`, then let inner search refine.
Persisting winners across tasks (the experience pool) is what the
frameworks do too — reuse a prior winner for a same-shape same-device op.

## Quick decision table

| Situation | First move |
| --- | --- |
| CPU, compute-bound | register-block (MR×NR), unroll 4, multiple accumulators |
| CPU, working set ≫ L1 | add one cache-blocking level; tile to fit L2 |
| GPU, any | channel-pack by 4 (vec4 texels), pad C to mult-of-4 |
| Constant weights | prepack once → free source → measure only the hot loop |
| Tile/workgroup unknown | seed from vendor table, then inner-search |

## Sources

`heuristics/parallelism_and_workgroup.md` (workgroup tables);
`references/frameworks.md` (full provenance & paths).
