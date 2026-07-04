# Heuristics — parallelism & workgroup sizing

Cross-framework rules for the parallelism knobs: CPU threading and GPU
workgroup shape. The GPU workgroup tables below are the highest-value
content here — they are ready-made **parameter seeds** for a vulkan
kernel's `local_size`, taken verbatim from shipping frameworks. Seed from
them, then let inner search refine.

## CPU threading

- One global scheduler / thread pool owns all parallelism; the kernel just
  sets a count. armnn clamps threads to **[1, 64]**, 0 = backend decides
  (`armnn/src/backends/neon/NeonWorkloadFactory.cpp:59-70`).
- **Single-threaded by default** — XNNPACK creates a pool only when
  `num_threads > 1` (`LiteRT/tflite/delegates/xnnpack/xnnpack_delegate.cc:652-654`).
- Work is split along **one window axis** per kernel (ACL `split_dimension`
  hint, `armnn/src/backends/neon/NeonInterceptorScheduler.cpp:28`). The
  kernel chooses the axis (usually the outermost independent one —
  channels / batch); the scheduler fans out.
- **Prior**: parallelize the outermost independent axis; skip threading
  when total work is small (launch/join dominates — the ncnn omp idiom
  and XNNPACK's single-thread default agree).

## GPU workgroup selection — vendor tables (verbatim seeds)

The reliable pattern: **hold total threads constant, shift mass toward Z
(the channel-slice axis) as channel count grows.**

**LiteRT GL default table** (product = 1024, keyed by `workload.z` =
channel slices, non-Mali) —
`LiteRT/tflite/delegates/gpu/gl/workgroups/default_calculator.cc:36-56`:

| z (slices) | local (x,y,z) |
| --- | --- |
| ≥64 | (4, 4, 64) |
| ≥32 | (8, 4, 32) |
| ≥16 | (8, 8, 16) |
| ≥8  | (16, 8, 8) |
| ≥4  | (16, 16, 4) |
| ≥2  | (32, 16, 2) |
| else | (32, 32, 1) |

**LiteRT Mali table** (product = 128 — Mali prefers smaller groups),
same file `:62-80`:

| z | local |
| --- | --- |
| ≥32 | (2, 2, 32) |
| ≥16 | (4, 2, 16) |
| ≥8  | (4, 4, 8) |
| ≥4  | (8, 4, 4) |
| ≥2  | (8, 8, 2) |
| else | (16, 8, 1) |

**ExecuTorch default picker** — sort axes descending, base
`{8, min(4,·), min(2,·)}`; degenerate: both trailing dims == 1 →
`{64,1,1}`; middle divisible by 4 → `{16,4,1}` else `{32,2,1}`
(`executorch/backends/vulkan/.../ComputeGraph.cpp:883-899`). Square
pickers: both spatial ≥6 → `{8,8,1}`; width < 6 → `{4,16,1}`; else
`{16,4,1}` (`Common.cpp:129-139`).

**Priors from these tables**:
- Default vulkan `local_size` for a channel-heavy 3D op ≈ **(8,8,16)** or
  the row matching the tensor's slice count; for a 1D/elementwise op ≈
  **(64,1,1)** or **(SUBGROUP_SIZE,1,1)**.
- Total invocations stay near the device max (1024 desktop-class, 128 Mali)
  — respect `MAX_WG_INVOCATIONS`.
- Mali wants **smaller** groups than Adreno/desktop; if targeting Mali,
  quarter the product.

## Two tuning modes (both real, both cached)

- **kExhaustive**: sweep candidate workgroups, time each with GPU events,
  cache the winner by kernel fingerprint
  (`LiteRT/tflite/delegates/gpu/cl/cl_operation.cc:195-223`). Candidates
  enumerated as grid divisors down to √n, or with slack up to
  `grid + 5` under `NO_ALIGNMENT` (`common/workgroup_selection.cc:135-141`).
- **kFast**: single heuristic pick — divider priority chain **8 > 4 > 2**
  for Z, halve grid for X, fill remaining budget on Y; conv caps XY at 256;
  fallback `{8,4,1}` (`LiteRT/.../work_group_picking.cc:148-224, 288`).
  Mali is forced to kFast (some drivers hang under a profiling queue).

**Prior for OptimizeAgent**: the workgroup knob is a small discrete set
(the table rows + a few divisors) — a perfect inner-search axis. Seed with
the table row for the tensor's slice count, expose 3–4 candidates around
it, let the search time them (that IS kExhaustive, scoped).

## Global dispatch size

Universal: `global = DivideRoundUp(grid, local) × local` (round grid up to
a workgroup multiple), then guard out-of-range invocations in the shader
(`LiteRT/.../gpu_operation.cc:35-60`;
`executorch/backends/vulkan/.../DynamicDispatchNode.cpp:46-49`). This is
why the dispatch-coverage guard in `vulkan/backend/idioms.md` is mandatory —
the round-up spawns dead lanes that must early-return.

## Sources

`vulkan/backend/idioms.md` (dispatch guard, workgroup declaration);
`bd_axes/*.md`; `references/frameworks.md`.
