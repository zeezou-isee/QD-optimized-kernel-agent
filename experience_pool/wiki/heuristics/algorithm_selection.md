# Heuristics — algorithm selection (compute-bound BD axis 1)

Cross-framework dispatch rules for choosing the algorithm family — the
anchor axis of the compute-bound BD coordinate system. These are the
"皇冠明珠" the design doc calls out: distilled `when-does-X-win` conditions,
generic across operators. Use them as priors for which family to try first,
not as a fixed decision — the search still measures.

## The conv/GEMM method space

ARM Compute Library exposes the method enum directly:
**`{DIRECT, GEMM, WINOGRAD, FFT}`**, with depthwise as a separate family
(`armnn/src/backends/neon/workloads/NeonWorkloadUtils.hpp:49-58`). The
choice is delegated to a heuristic keyed on a **feature vector**:

> method = f(input shape, weight shape, output shape, stride, dilation,
> dtype, fused-activation, fast_math flag, target-arch)

(`armnn/src/backends/neon/workloads/NeonConvolution2dWorkload.cpp:118-126`;
GPU adds device model: `armnn/src/backends/cl/workloads/ClConvolution2dWorkload.cpp:135-144`).

**Prior**: when the operator is conv-like and compute-bound, treat these
seven features as the inputs that decide the algorithm-family cell. Two
proposals that differ only in an irrelevant feature land in the same cell.

## When each family wins (cross-framework priors)

| Family | Wins when | Evidence |
| --- | --- | --- |
| direct | small `num_input × num_output`; arbitrary kernel/stride/dilation | ncnn `convolution_arm.cpp` dispatch fallback |
| im2col-GEMM | GEMM shape favorable (large `out_h*out_w`); dynamic or large channels | ncnn; ACL GEMM method |
| winograd (F(2,3)/F(4,3)/F(6,3)) | **3×3, stride 1, dilation 1**, enough channels | ncnn `convolution_arm.cpp:204-262`; ACL |
| depthwise (specialized) | depthwise conv; separate 3×3/5×5 assembly | ncnn; ACL `NeonDepthwiseConvolutionWorkload` |
| 1×1 | pointwise; degenerates to GEMM | ncnn; executorch `conv2d_pw` |
| cooperative-matrix | large GEMM on capable GPU only — **NOT mobile** | executorch `GemmCoopmat.h` (see below) |

## Winograd requires a precision opt-in

armnn only selects Winograd when `fast_math` is enabled — a bit-exact
model never gets Winograd, because its transforms lose fp32 exactness
(`armnn/src/backends/neon/test/NeonCreateWorkloadTests.cpp:281-297`). It
can also be force-disabled via a `DisableWinograd` option
(`armnn/docs/05_04_runtimeoptions.dox:112`).

**Prior**: propose Winograd only when the tolerance budget allows ~1-2
bits of relative error per pass; stacking Winograd layers compounds it. In
`memory_bound` regime, don't propose it at all (it cuts FLOPs, not bytes).

## Constant vs dynamic operands select different kernels

A GEMM with **constant** weights takes the prepack path; a GEMM with
**dynamic** operands (both inputs runtime tensors) takes an on-the-fly
kernel — "GeMM dispatches kernel handles dynamic inputs differently"
(`armnn/src/backends/neon/workloads/NeonBatchMatMulWorkload.cpp:59-61`).

**Prior**: check whether the second operand is a constant weight or a
runtime activation before choosing the GEMM variant.

## Cooperative-matrix / tensor-core gate (skip on mobile)

ExecuTorch gates its coopmat GEMM behind (verbatim,
`executorch/backends/vulkan/.../GemmCoopmat.h:39-53`):

```
dim(out) <= 2 && supports_cooperative_matrix() && subgroup_size() == 64
  && !is_integrated_gpu() && storage == kBuffer
  && M % 64 == 0 && N % 64 == 0 && K % 32 == 0
```

Tiles are `M=N=64, K=32, 256 invocations`; it is **explicitly
desktop-tuned and disabled on integrated / mobile GPUs**, with no
partial-tile handling.

**Prior**: on Apple / mobile GPUs (`HAS_COOPMAT == 0`, integrated), do NOT
propose cooperative-matrix — it will be rejected or absent. Use `vec4` /
subgroup mapping instead.

## Fast-path shape gates (from XNNPACK partitioning — verbatim)

XNNPACK refuses ops outside its fast envelope — these thresholds tell you
what the SoTA CPU library considers cheaply optimizable:
- conv **`stride > 2` → not accelerated** (`executorch/backends/xnnpack/.../gemm_configs.py:364`)
- dynamic-quant conv must be **exactly 2D, non-depthwise** (`:376-378`)
- add/sub `alpha` must be `isclose(1.0, atol=1e-9)` (`generic_node_configs.py:140`)
- softmax only on the **last dim** (`:301`)
- slice **stride must be 1** (`:632`)
- mean reduction: rank 4, dims `[2,3]`, keepdim (`:541-568`)

**Prior**: these are the "easy" cases. An op outside them (stride>2 conv,
non-last-axis softmax, strided slice) is a harder cell where a custom
kernel has more to gain but also more ways to be wrong.

## Sources

`bd_axes/compute_bound.md` (the axis this feeds); `references/frameworks.md`
(paths). Availability flags: `arm/hardware/*.json`, `vulkan/hardware/*.json`.
