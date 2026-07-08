# Mobilekernelbench_optimized — 30 ops for OptimizeAgent (v3)

Curated 30-op subset for OptimizeAgent. **v3**: v2 dropped 12 below-noise-floor ops (TopK/Det/Gather/GridSample/Softmax/Concat/ScatterElements/ArgMax/Sub/Celu/MaxPool_2d_ceil/StridedSlice); v3 also replaced Einsum (crashed on arm — Cand_Einsum_arm packed-path kernel limit) with Group_Convolution_2D_kernel. All 30 now run on arm and measure above the device noise floor.

Priorities: 1) slower than ncnn native, 2) hard ops, 3) diversity.

- slower than native (speedup_fair<1): 17/30

- categories: Convolution×12, Matrix×3, Normalization×3, Tensor×3, Trigonometry×3, Unary×3, Reduction×2, Activation×1


> Decomposed ops excluded (QD winner never lands — audit_decomposed_ops.py).

> Optimize results (rounds + real-phone speedup) roll up to batch/results/optimize_rollup.md.


`speedup_fair`=native/ours (fp32); <1 ⇒ slower. `ours_ms`=our fp32 min latency (arm perf sweep).


| # | op | category | ncnn layer | ours_ms | speedup_fair | note |
|---|----|----------|-----------|--------:|-------------:|------|
| 1 | `Dense_Convolution_2D` | Convolution | Convolution | 406.42 | 0.010 | kept from v1 |
| 2 | `Group_Convolution_2D` | Convolution | ConvolutionDepthWise | 55.13 | 0.036 | kept from v1 |
| 3 | `Gemm_no_bias` | Matrix | Gemm | 1.06 | 0.047 | kept from v1 |
| 4 | `Conv` | Convolution | Convolution | 4.27 | 0.075 | slow direct conv (sf 0.075, 4.3ms) |
| 5 | `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | Deconvolution | 276.17 | 0.080 | kept from v1 |
| 6 | `Winograd_Convolution_2D_padding` | Convolution | Convolution | 61.53 | 0.084 | slow+hard: padded Winograd (sf 0.084, 61ms) |
| 7 | `Strassen_Convolution_2D` | Convolution | Convolution | 38.09 | 0.090 | kept from v1 |
| 8 | `Winograd_Convolution_2D` | Convolution | Convolution | 734.85 | 0.111 | kept from v1 |
| 9 | `Group_Convolution_2D_kernel` | Convolution | ConvolutionDepthWise | 20.54 | 0.134 | slow+hard: grouped conv (kernel variant, replaces crash Einsum; sf 0.134, 20ms) |
| 10 | `ConvTranspose_dilations` | Convolution | Deconvolution | 31.52 | 0.234 | kept from v1 |
| 11 | `Einsum_sum_all` | Matrix | Reduction | 0.02 | 0.500 | kept from v1 |
| 12 | `Floor` | Unary | UnaryOp | 0.34 | 0.676 | kept from v1 |
| 13 | `Exp` | Unary | UnaryOp | 2.35 | 0.800 | kept from v1 |
| 14 | `Cos` | Trigonometry | UnaryOp | 1.09 | 0.835 | kept from v1 |
| 15 | `Tan` | Trigonometry | UnaryOp | 1.84 | 0.875 | slow unary trig (sf 0.875, 1.8ms) |
| 16 | `Sinh` | Trigonometry | UnaryOp | 9.00 | 0.953 | slow heavy unary (sf 0.953, 9ms) |
| 17 | `Round` | Unary | UnaryOp | 2.08 | 0.995 | slow unary round (sf 0.995, 2.1ms) |
| 18 | `InstanceNormalization` | Normalization | InstanceNorm | 0.83 | 1.000 | kept from v1 |
| 19 | `CumSum` | Tensor | CumulativeSum | 0.01 | 1.000 | kept from v1 |
| 20 | `Conv_with_strides_padding` | Convolution | Convolution | 1.24 | 1.040 | strided conv config (1.2ms) |
| 21 | `DepthToSpace` | Tensor | PixelShuffle | 0.14 | 1.357 | kept from v1 |
| 22 | `LayerNorm` | Normalization | LayerNorm | 0.35 | 1.771 | kept from v1 |
| 23 | `Softplus_3d` | Activation | Softplus | 0.75 | 2.013 | softplus activation (0.75ms) |
| 24 | `BatchNormalization` | Normalization | BatchNorm | 0.72 | 2.097 | batchnorm (0.72ms, new norm type) |
| 25 | `Clip` | Tensor | Clip | 1.58 | 2.158 | clip activation (1.6ms) |
| 26 | `ReduceMean` | Reduction | Reduction | 2.71 | 6.963 | reduction, ms-scale (2.7ms) |
| 27 | `DeconvolutionDepthwise_2D_stride` | Convolution | DeconvolutionDepthWise | 23.34 | 7.855 | deconv-depthwise, heavy (23ms) |
| 28 | `ReduceMax_with_negative_values` | Reduction | Reduction | 2.19 | 8.813 | reduction variant (2.2ms) |
| 29 | `Conv3D` | Convolution | Convolution3D | — | — | kept from v1 |
| 30 | `MatMul` | Matrix | MatMul | — | — | kept from v1 |
