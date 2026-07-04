# OptimizeAgent Wiki (v1 — generic)

Prescriptive knowledge base injected into OptimizeAgent's LLM proposer
prompts. Encodes the **methodology of kernel optimization** (primitives,
BD search axes, regime rules, backend language) — NOT per-operator recipes.
Operator-agnostic and mostly framework-agnostic.

## Design

Four cross-cutting layers + two backend-specific layers.

```
primitives/                        ← optimization taxonomy (algorithm-agnostic)
  reduce_compute.md                  moves that cut FLOPs
  reduce_memory_traffic.md           moves that cut bytes
  increase_parallelism.md            moves that expose parallelism / hide latency
  hardware_specialized.md            moves that use dedicated instructions

bd_axes/                           ← MAP-Elites search-space definition per regime
  memory_bound.md                    layout × tiling × fusion coordinate system
  compute_bound.md                   algorithm × mapping coordinate system
  mixed.md                           when to blend / default to memory_bound

heuristics/                        ← cross-framework "where to look first" priors
  tiling_and_packing.md              register/cache blocking, channel-pack-4, prepack
  algorithm_selection.md             conv/gemm method dispatch conditions
  precision_and_quant.md             fp16/int8 rules, fp32-accumulate, requant
  parallelism_and_workgroup.md       CPU threading + GPU workgroup seed tables

bottleneck/roofline_regimes.md     ← regime classification, early-stop rules

references/frameworks.md           ← provenance map (NOT injected)

arm/                               ← ARM CPU backend
  hardware/apple_m5.json             hw_ns extras (extends ConstraintEngine namespace)
  backend/dialect.md                 NEON intrinsics reference
  backend/idioms.md                  ncnn Mat/cstep/omp/tail conventions
  backend/failure_patterns.md        E1..E6 preemption

vulkan/                            ← ncnn Vulkan backend
  hardware/apple_m5.json             hw_ns extras (Tier-2 device profile)
  backend/dialect.md                 ncnn GLSL shader dialect (sfp/afp/psc/…)
  backend/idioms.md                  workgroup guard, dispatch coverage, MoltenVK
  backend/failure_patterns.md        E1..E8 preemption
```

## Retrieval contract

`opgen/optimize/proposer/wiki.py::WikiLoader.context_block(regime)` returns
a fixed block:

1. All 4 primitive pages
2. `bd_axes/<regime>.md` (routed by `regime`; falls back to `mixed` for
   unknown / empty)
3. All 4 `heuristics/` pages (cross-framework search priors)
4. `bottleneck/roofline_regimes.md`
5. `backend/dialect.md`
6. `backend/idioms.md`
7. `backend/failure_patterns.md`

`references/` is NOT injected (provenance for maintainers only).

Regime comes from `policy/roofline.py::diagnose()` (already computed for
MAP-Elites; for M1 linear the OptimizeAgent computes it from the naive
static AI). Unknown regime → wiki treats as `mixed` and hedges.

`WikiLoader.hardware_extras(hw_key)` also merges backend-scoped symbols
into `ConstraintEngine.hw_ns` (arm: L3/CACHE_LINE/HAS_DOTPROD/…; vulkan:
SUBGROUP_SIZE/MAX_SHARED_MEM_BYTES/HAS_FP16/…) so the LLM's constraint
expressions can reference them.

## Authoring rules

- **Do not write per-operator content.** If a page names Convolution or
  BinaryOp specifically, it does not belong here — put it in the ncnn
  layer-interface dictionary instead.
- **Do not copy content from `opgen/kernel/kernel_prompts.py`.**
  KernelAgent writes baselines; OptimizeAgent optimizes them. Cross-reference,
  don't duplicate.
- **All-caps identifiers must be declared symbols.** The LLM uses them in
  constraint expressions. New symbols must land in `ConstraintEngine.hw_ns`
  (via `WikiLoader.hardware_extras`); `WikiLoader._check_symbols()` warns
  on drift.
- **Distill, don't paste.** Cite `ncnn:src/layer/…:Lxxx`; do not vendor source.
- **Length caps** (enforced by the loader; overflow is truncated):
  - primitive page: 250 lines
  - bd_axes page: 250 lines
  - heuristic page: 200 lines
  - roofline_regimes: 150 lines
  - dialect: 400 lines
  - idioms: 200 lines
  - failure_patterns: 80 lines
  - total per prompt: 3600 lines

## Source

Design references under `AgentDesign/prologue/`:
- `算子优化-问题建模与体系设计.md` §1 (optimization taxonomy),
  §5.1 (roofline triple role), §8.3-4 (experience-pool schema)
- `算子优化-完整Workflow.md` §4.1 (regime classification),
  §4.2 (BD coordinate systems), §5.4 (failure taxonomy),
  §8.2 (early-stop conditions)
- `微观参数优化设计.md` (axis classification: structural vs refinement)

Cross-framework grounding (local clones under `kernelgen/`): ncnn (primary),
armnn/ACL, executorch, LiteRT/XNNPACK. Full provenance +
per-page source map in `references/frameworks.md`.
