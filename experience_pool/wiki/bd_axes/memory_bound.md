# BD coordinate system — memory-bound regime

The **Behavior Descriptor (BD) coordinate system** defines the search
space's structural axes for MAP-Elites. Each proposal is classified into
one cell in this grid, and MAP-Elites keeps the best kernel per cell so
the archive covers diverse strategies rather than converging on one point.

**Regime lock**: this coordinate system is used when
`regime == memory_bound` (the kernel's static AI < hardware ridge point).
The main lever is bytes moved / access pattern. Algorithm-family axis is
NOT a main axis here — reducing FLOPs on a memory-bound kernel does not
reduce wall time.

Two types of axes:
- **Structural axes** (positioning): known at generation time from the
  proposal's structure. These decide which cell the proposal lives in.
- **Refinement axes** (post-hoc): only measurable after inner search fixes
  parameters. These subdivide within a cell.

## Structural axes (positioning)

### Axis 1 (anchor) — data-layout / access-pattern family

The choice of memory layout dominates memory-bound performance. Category
values are backend-scoped:

| Backend | Layout categories |
| --- | --- |
| **ARM CPU** | `{NCHW, NHWC, NC4HW4-packing}` |
| **Vulkan** | `{buffer × NCHW, buffer × NHWC, buffer × NC4HW4, image × NCHW, image × NHWC, image × NC4HW4}` |

The `{buffer, image}` sub-choice on Vulkan matters: image storage has
hardware texture-cache locality but restricted access patterns; buffers
are general but rely on driver cache.

### Axis 2 — tiling-strategy family

How aggressively the kernel partitions its working set:

| Category | Meaning |
| --- | --- |
| `no_tiling` | one pass over the tensor, relies on hardware prefetchers |
| `single_level` | one nested tile (usually L1 or GPU shared) |
| `two_level_plus_register` | register-blocked innermost + one cache-level tile |

Cross-product with Axis 1 gives roughly 9–18 cells depending on backend;
this is the target coverage size for MAP-Elites in memory-bound regime.

### Axis 3 (optional) — fusion degree

Merging adjacent ops kills intermediates:

| Category | Meaning |
| --- | --- |
| `no_fusion` | this kernel does one op only |
| `one_fused` | one adjacent op absorbed (element-wise chain, bias, activation) |
| `two_or_more_fused` | multi-op fusion (e.g. reduction + activation) |

Enable this axis only when the search is authorized to fuse (the operator
has a known adjacent partner in the model).

## Refinement axes (post-hoc, after inner search)

### Working-set cache tier (memory-bound-specific)

| Category | Meaning |
| --- | --- |
| `fits_in_L1` | working set ≤ `L1D` (ARM) / ≤ `MAX_SHARED_MEM_BYTES` (Vulkan) |
| `fits_in_L2` | working set ≤ `L2` (ARM) or fits in an on-chip cache |
| `spills` | working set > cache — bandwidth-limited by global memory |

Computed after inner search picks concrete tile sizes.

### Arithmetic-intensity band (shared with compute-bound)

`AI / ridge_point` normalized: `{ < 0.5, 0.5–1.0, > 1.0 }`. Post-hoc,
never drives pre-search positioning.

## Rules

- Structural axes are set **before** any real measurement, purely from the
  proposal's declared `techniques` / layout choice. They gate which cell
  competes with which.
- Refinement axes are set **after** inner search finishes, from the
  measured tile / occupancy. They can subdivide a cell if the archive is
  configured for it, but never rewrite cell membership.
- A parameter-level failure (OOM at one tile size) shrinks the constraint
  region for that template; it does NOT invalidate the cell.
- A structural failure (template can't compile at all) blocks the whole
  structural coordinate — this becomes a **global negative constraint**
  the proposer must avoid in future rounds.

## What primitives fit this regime

Primary levers (see `primitives/reduce_memory_traffic.md`):
- Tiling / blocking — sets Axis 2
- Data-layout transforms — sets Axis 1
- Packing (`NC4HW4`, `sfpvec4/8`) — sets Axis 1
- Fusion of intermediates — sets Axis 3

Secondary (see `primitives/increase_parallelism.md`):
- Prefetching, double buffering — hide the latency memory-bound kernels
  can't avoid
- Vectorization — required for lane-parallel loads, but not a memory move

Rarely useful here (see `primitives/reduce_compute.md`):
- Algorithm substitution (Winograd, FFT) — cuts FLOPs, not bytes
- Precision reduction — DOES help (halves bytes); it's the one
  compute-family primitive with memory-bound leverage

## Source

`AgentDesign/prologue/算子优化-完整Workflow.md:99-146` (BD coordinate systems);
`微观参数优化设计.md` (axis classification into structural vs refinement).
