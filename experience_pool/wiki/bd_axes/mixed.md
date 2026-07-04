# BD coordinate system — mixed regime

When the kernel's static AI is close to the hardware ridge point (roughly
within ±30%), a single coordinate system is misleading. The kernel may
tip either way depending on the specific optimization applied. Two
strategies:

## Strategy 1: default to memory-bound

Conservative choice. Reasoning:
- On mobile hardware ridge points are typically low (memory bandwidth
  scarce), so an AI near the ridge usually means memory-bound in
  practice under any real access pattern (cache misses push effective AI
  DOWN, not up).
- Memory-bound primitives (layout, tiling, packing) also apply to
  compute-bound kernels — they just yield less.
- Compute-bound-only primitives (algorithm substitution) can be tried
  from within the memory-bound coordinate system as a "compute variant"
  proposal that lives in an Axis 3 = fusion cell.

Use this when: unknown operator, first attempt, or the AI estimate is
noisy (e.g. compiled with unknown fusion behavior).

## Strategy 2: dual-regime search

Run two shorter MAP-Elites arms in parallel, one per coordinate system,
then union the archives. Reasoning:
- The regime-locking rule is a simplification. Some operators genuinely
  benefit from both memory and compute optimizations independently.
- Doubling the budget across two regime-locked runs is often cheaper
  than one long unfocused run.

Use this when: the AI estimate falls in [0.7 × ridge, 1.4 × ridge], the
operator is well-known, and you have budget headroom.

## Axis choice for mixed

Prefer memory-bound coordinate axes as the primary layout, with algorithm
family included as an optional Axis 4 (rather than the primary Axis 1 it
would be in compute-bound). This lets the archive record both
"layout-driven" and "algorithm-driven" winners in adjacent cells.

## When to promote/demote

If measurements consistently show the kernel is bandwidth-limited even
after best-effort layout + tiling, demote to `memory_bound` and lock the
coordinate system.

If measurements consistently show the kernel is compute-limited even
after best-effort algorithm substitution, promote to `compute_bound` and
lock the coordinate system.

The regime lock happens once per problem (operator + shape + hardware);
mid-search reclassification is expensive because the archive shape
changes.

## Source

`AgentDesign/prologue/算子优化-完整Workflow.md:88-98` (roofline & regime
locking); `算子优化-问题建模与体系设计.md:167-176` (瓶颈会随优化翻转).
