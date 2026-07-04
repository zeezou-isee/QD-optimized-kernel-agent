# Roofline regimes — classification, signals, and early stop

The roofline model gives OptimizeAgent three things:
1. **A ceiling** — theoretical peak throughput (compute or bandwidth wall)
2. **A current position** — where the measured kernel sits vs the ceiling
3. **A regime lock** — which BD coordinate system to search in, and which
   primitives are likely to move the needle

## Regime classification

**Inputs**:
- `AI = total_FLOP / total_bytes` computed on the **naive implementation**
  of the operator (independent of any specific optimization). This is a
  problem-level property of `(operator, shape, dtype)`.
- `ridge_point = peak_compute / peak_bandwidth` from the hardware
  profile.

**Rule**:
```
if AI <  ridge_point:  regime = memory_bound
if AI >= ridge_point:  regime = compute_bound
if 0.7 * ridge_point <= AI <= 1.4 * ridge_point:  regime = mixed  (optional)
```

Both `AI` and `ridge_point` are computed **once per problem** and locked
for the run. Do not reclassify mid-search — the archive shape depends on
the choice.

## Regime → coordinate system → search moves

| Regime | Coordinate system | Primary primitives |
| --- | --- | --- |
| memory_bound | `bd_axes/memory_bound.md` (layout × tiling × fusion) | `reduce_memory_traffic.md` + fusion + precision reduction |
| compute_bound | `bd_axes/compute_bound.md` (algorithm × mapping) | `reduce_compute.md` (algorithm substitution) + `hardware_specialized.md` |
| mixed | `bd_axes/mixed.md` (layout-primary + algorithm secondary) | union, with layout as default |

Do NOT propose:
- **Memory-bound regime + algorithm substitution** (Winograd on a small
  channel-count conv) — it moves the roofline ridge but the kernel
  stays memory-bound.
- **Compute-bound regime + layout transform** — the kernel isn't
  stalling on memory; the transform costs a pass with no benefit.

## Regime can flip at kernel level

The design doc explicitly notes: "瓶颈是实现相关的、会随优化翻转" — a naive conv
may be memory-bound, but after sufficient tiling it becomes compute-bound.
This shift **does not change the locked coordinate system** — it is
absorbed by the shared AI post-hoc refinement axis. If measured AI is
consistently far from the naive AI after optimization, that's a signal
the regime lock was wrong; the correct response is to end the run and
re-classify.

## Roofline as early-stop condition

Three stop conditions (any one triggers termination of the outer search):

1. **Roofline proximity**: `best_latency / theoretical_min_latency < 1 + epsilon`
   (typically `epsilon = 0.05`). If the kernel is within 5% of the
   theoretical peak, additional search is unlikely to pay for itself.
2. **Budget exhausted**: total measurements ≥ configured budget (e.g. 80
   in the current MAP-Elites default).
3. **Convergence stall**: the global best latency has not improved by
   more than `epsilon` for K consecutive rounds (patience). Signals
   diminishing returns.

Roofline proximity is the "luxury" stop — most searches terminate on
budget or patience first.

## Constraint-equation vocabulary (used by proposals)

The proposer writes physical constraints referring to hardware symbols.
For LLM constraints to actually gate anything, both sides must agree on
the symbol names.

Common symbols (see `arm/hardware/*.json` and `vulkan/hardware/*.json`
for the exact set per device):

| Symbol | Meaning | Regime that cares |
| --- | --- | --- |
| `L1D`, `L2`, `L3` | cache sizes (bytes) | memory_bound (tile ≤ Lx) |
| `VEC_BITS`, `FP32_PER_VEC`, `VECTOR_REGS` | SIMD width & register budget | compute_bound (unroll ≤ VECTOR_REGS) |
| `CACHE_LINE` | alignment / prefetch stride | memory_bound |
| `SUBGROUP_SIZE`, `MAX_WG_INVOCATIONS` | GPU workgroup limits | vulkan (any regime) |
| `MAX_SHARED_MEM_BYTES` | shared memory tile bound | memory_bound (shared_tile_bytes ≤ MAX_SHARED_MEM_BYTES) |
| `HAS_DOTPROD`, `HAS_ASIMDHP`, `HAS_FP16`, `HAS_COOPMAT` | feature flags | availability guards |

Rule: a constraint that references a symbol not in the current backend's
namespace is silently dropped by `ConstraintEngine` (safer than rejecting
the whole template). Prefer symbols listed in the hardware block of the
prompt.

## Source

`AgentDesign/prologue/算子优化-完整Workflow.md:88-98` (regime classification);
`算子优化-完整Workflow.md:425-427` (three stop conditions);
`算子优化-问题建模与体系设计.md:167-176` (roofline's triple role);
`算子优化-问题建模与体系设计.md:400-407` (hardware constants schema).
