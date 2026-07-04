# Framework provenance map (authoring reference — NOT injected)

Where the wiki's knowledge comes from, and which repo to consult when
extending it. All repos are cloned as siblings of the agent repo under
`/Users/xingze/Documents/project/kernelgen/`. This page is for wiki
maintainers; it is NOT injected into prompts (the loader skips
`references/`).

## Repos

| Repo | Origin | Role in the wiki |
| --- | --- | --- |
| `ncnn` | Tencent/ncnn | **Primary.** All backend dialect + idioms + failure patterns. The target framework: kernels are generated FOR ncnn. |
| `armnn` | ARM-software/armnn | ARM conv-method selection, CLTuner autotune, weight prepack, NHWC policy, thread clamp. ACL micro-kernel *source* is not bundled (headers only) — armnn exposes the selection layer. |
| `executorch` | pytorch/executorch | Vulkan GLSL codegen (template+yaml), spec-const local size, channel packing, coopmat gate, cortex_m int8 requant, ATen-based optimized CPU kernels. |
| `LiteRT` | google-ai-edge/LiteRT | XNNPACK integration (MR×NR + weight cache), GPU delegate workgroup tables + texture/buffer decision + PHWC4 layout, quant tradeoff docs. XNNPACK C source fetched at build (not vendored). |

## What each wiki page draws from

| Wiki page | Sources |
| --- | --- |
| `primitives/*` | design doc taxonomy (`AgentDesign/prologue/`), grounded by all 4 frameworks |
| `bd_axes/*` | design doc §4.2/§4.4 (BD coordinate systems) |
| `bottleneck/roofline_regimes.md` | design doc §4.1/§5.1/§8.2 |
| `heuristics/tiling_and_packing.md` | XNNPACK MR×NR, executorch BLAS, armnn prepack, LiteRT PHWC4 + weight cache |
| `heuristics/algorithm_selection.md` | armnn `get_convolution_method`, ncnn conv dispatch, executorch coopmat gate, XNNPACK partitioning thresholds |
| `heuristics/precision_and_quant.md` | executorch cortex_m requant, LiteRT precision modes + tradeoff table, XNNPACK fp16 tiers |
| `heuristics/parallelism_and_workgroup.md` | LiteRT GL/Mali workgroup tables, executorch workgroup pickers, armnn/XNNPACK threading |
| `arm/backend/*` | ncnn `src/layer/arm/*`; ACL selection layer via armnn |
| `vulkan/backend/*` | ncnn `src/layer/vulkan/shader/*.comp`; executorch GLSL graph; LiteRT GPU delegate |
| `{arm,vulkan}/hardware/*.json` | local `sysctl` (arm), vulkaninfo Tier-2 profile (vulkan) |

## Key goldmine paths (for future distillation)

- ncnn: `ncnn/src/layer/arm/`, `ncnn/src/layer/vulkan/shader/*.comp`
- armnn: `armnn/src/backends/{neon,cl}/workloads/`,
  `armnn/src/backends/aclCommon/ArmComputeTuningUtils.hpp`,
  `armnn/docs/05_04_runtimeoptions.dox`
- executorch: `executorch/backends/vulkan/runtime/graph/ops/glsl/`,
  `executorch/backends/vulkan/runtime/graph/ComputeGraph.cpp`,
  `executorch/kernels/optimized/blas/BlasKernel.cpp`,
  `executorch/backends/cortex_m/passes/`
- LiteRT: `LiteRT/tflite/delegates/gpu/` (`common/task/`, `gl/workgroups/`,
  `cl/`), `LiteRT/tflite/delegates/xnnpack/README.md`

## Distillation rules (keep the wiki generic)

- Extract **method and decision rules**, never per-operator recipes.
- Quote heuristic **constants and tables verbatim** with `repo:file:line` —
  those are the highest-value, lowest-ambiguity content.
- Cross-framework agreement → strong prior. Divergence → a device/shape
  dependency worth a search axis; record both, don't pick.
- Do not vendor source into the wiki; cite paths.

## Web access

Web search / fetch were blocked in the environment where this wiki was
built (proxy does not expose web_search; domain verification failed). All
content above is from the local clones, which ARE the public repos. If web
becomes available, ARM's "Neon Programmer's Guide", the Vulkan spec's
compute chapter, and framework perf docs are the natural next sources.
