# BD coordinate system — compute-bound regime

**Regime lock**: this coordinate system is used when
`regime == compute_bound` (the kernel's static AI ≥ hardware ridge point).
The main lever is FLOP / throughput. Data-layout is NOT a main axis here —
reshuffling bytes doesn't help a kernel that's already saturating memory.

Two types of axes:
- **Structural axes** (positioning): known at generation time from the
  proposal's structure. These decide which cell the proposal lives in.
- **Refinement axes** (post-hoc): only measurable after inner search fixes
  parameters.

## Structural axes (positioning)

### Axis 1 (anchor) — algorithm family

The choice of algorithm dominates compute-bound performance. Category
values are drawn from a small, well-known catalog (per operator class the
menu differs, but the axis type is universal):

| Example category set (for conv) |
| --- |
| `{direct, im2col_gemm, winograd23, winograd43, winograd63, fft, depthwise, 1x1}` |

For other operators, the menu shrinks: element-wise has essentially one
algorithm; matmul has `{direct, blocked, strassen}`; reduction has
`{linear, tree, subgroup-fast-path}`. What matters is **discreteness** —
a small enumerated menu the proposer picks from.

### Axis 2 — compute-mapping / instruction class

How the algorithm is lowered to hardware compute:

| Backend | Mapping categories |
| --- | --- |
| **ARM CPU** | `{scalar, NEON-vectorized, dotprod/sdot, sme2}` |
| **Vulkan** | `{scalar, vec4, sfpvec8, cooperative-matrix}` |

Availability is device-scoped: the `dotprod` category is legal only if
`HAS_DOTPROD == 1`, `cooperative-matrix` only if `HAS_COOPMAT == 1`, etc.
Guard at proposal time; don't propose a category the device doesn't have.

## Refinement axes (post-hoc, after inner search)

### Occupancy / ILP tier

| Category | Meaning |
| --- | --- |
| `low` | few in-flight ops, latency-exposed |
| `medium` | balanced |
| `high` | many independent chains, throughput-bound |

Measured after inner search fixes unroll factors and thread counts.

### Arithmetic-intensity band (shared with memory-bound)

`AI / ridge_point` normalized: `{ < 0.5, 0.5–1.0, > 1.0 }`. Post-hoc,
never drives pre-search positioning. Cross-regime comparison signal only.

## Rules

- Structural axes are set **before** measurement from the proposal's
  declared `techniques` / algorithm choice.
- Refinement axes are set **after** inner search, from the measured
  performance profile.
- The `algorithm × mapping` product is small (typically 4–8 × 3–4 = 12–32
  cells) — MAP-Elites should be able to cover most of it within budget.
- Availability constraint: any (algorithm, mapping) cell whose hardware
  requirement is absent is a **structurally infeasible** cell; the
  proposer should not propose it.

## What primitives fit this regime

Primary levers (see `primitives/increase_parallelism.md` and
`primitives/hardware_specialized.md`):
- Vectorization — sets Axis 2 category
- Multi-level parallel mapping — sets Axis 2 category
- Specialized MAC instructions (dotprod, cooperative-matrix) — sets Axis 2
- Loop unroll + multiple accumulators — pushes occupancy/ILP refinement

Also useful (see `primitives/reduce_compute.md`):
- Algorithm substitution — sets Axis 1 category (this is the biggest
  single move in compute-bound regime)
- Precision reduction — moves the ridge point favorably AND enables new
  Axis 2 categories (fp16 arithmetic, int8 dotprod)

Rarely useful here (see `primitives/reduce_memory_traffic.md`):
- Layout transforms — the compute-bound kernel isn't stalling on memory
- Tiling — register-blocking still matters (Axis 2 accumulator budget),
  but broader cache tiling is a wash

## Source

`AgentDesign/prologue/算子优化-完整Workflow.md:99-146` (BD coordinate systems);
`微观参数优化设计.md` (axis classification into structural vs refinement).
