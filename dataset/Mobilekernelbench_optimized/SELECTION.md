# Mobilekernelbench_optimized — 30 ops for OptimizeAgent (v4)

**v4**: restructured to a **2:1 slower:other** split, all ms-scale (sandbox device baseline measured on a stable single phone).

- **20 slower-than-ncnn** (perf `speedup_fair`<1 vs ncnn native) — the QD headroom set

- **10 other** (at/faster than native; hard/diverse) — the control set

- **all measured on-device above the noise floor**; 16 slower ops are ≥0.3ms, 4 fillers (Group_Convolution_2D_kernel/Cosh/Atanh/Conv) sit at ~0.19–0.26ms (still ~10× the 0.02ms floor) because the dataset has only 16 ms-scale slower ops.


categories: Convolution×15, Trigonometry×7, Unary×3, Reduction×3, Tensor×1, Matrix×1 — conv/trig heavy by necessity (ms-scale *slower* ops are dominated by conv & transcendental families; smaller ops fall below the device noise floor).


> `sandbox_ms` = on-device baseline latency (avg/100) in the optimize sandbox — the figure that decides ms-scale. `speedup_fair` = ncnn-native/ours (fp32 whole-net).


## Slower-than-ncnn (20)

| # | op | category | ncnn layer | sandbox_ms | speedup_fair |
|---|----|----------|-----------|----------:|-------------:|
| 1 | `Dense_Convolution_2D_kernel` | Convolution | Convolution | 223.3389 | 0.149 |
| 2 | `Winograd_Convolution_2D` | Convolution | Convolution | 25.0174 | 0.111 |
| 3 | `ConvTranspose_dilations` | Convolution | Deconvolution | 13.9396 | 0.234 |
| 4 | `Dense_Convolution_2D` | Convolution | Convolution | 9.4377 | 0.01 |
| 5 | `Sinh` | Trigonometry | UnaryOp | 5.617 | 0.953 |
| 6 | `Asin` | Trigonometry | UnaryOp | 4.303 | 0.938 |
| 7 | `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | Deconvolution | 3.9111 | 0.08 |
| 8 | `Round` | Unary | UnaryOp | 3.1966 | 0.995 |
| 9 | `Exp` | Unary | UnaryOp | 2.8279 | 0.8 |
| 10 | `Group_Convolution_2D` | Convolution | ConvolutionDepthWise | 2.2345 | 0.036 |
| 11 | `Cos` | Trigonometry | UnaryOp | 1.7598 | 0.835 |
| 12 | `Winograd_Convolution_2D_padding` | Convolution | Convolution | 1.4619 | 0.084 |
| 13 | `Floor` | Unary | UnaryOp | 1.335 | 0.676 |
| 14 | `Strassen_Convolution_2D` | Convolution | Convolution | 1.1576 | 0.09 |
| 15 | `Tan` | Trigonometry | UnaryOp | 1.0255 | 0.875 |
| 16 | `Asinh` | Trigonometry | UnaryOp | 0.3905 | 0.829 |
| 17 | `Group_Convolution_2D_kernel` | Convolution | ConvolutionDepthWise | 0.2635 | 0.134 |
| 18 | `Cosh` | Trigonometry | UnaryOp | 0.2407 | 0.932 |
| 19 | `Atanh` | Trigonometry | UnaryOp | 0.2087 | 0.882 |
| 20 | `Conv` | Convolution | Convolution | 0.1854 | 0.075 |

## Other (10)

| # | op | category | ncnn layer | sandbox_ms | speedup_fair |
|---|----|----------|-----------|----------:|-------------:|
| 1 | `Conv3D` | Convolution | Convolution3D | 38.8112 | None |
| 2 | `Clip` | Tensor | Clip | 7.1187 | 2.158 |
| 3 | `Dense_Convolution_1D` | Convolution | Convolution1D | 2.2364 | 1.737 |
| 4 | `DeconvolutionDepthwise_2D_kernel` | Convolution | DeconvolutionDepthWise | 1.2339 | 2.918 |
| 5 | `ReduceMin_with_negative_values` | Reduction | Reduction | 1.214 | 8.818 |
| 6 | `DeconvolutionDepthwise_2D_stride` | Convolution | DeconvolutionDepthWise | 1.1501 | 7.855 |
| 7 | `ReduceMean` | Reduction | Reduction | 1.0356 | 6.963 |
| 8 | `ReduceMax_with_negative_values` | Reduction | Reduction | 1.019 | 8.813 |
| 9 | `Deconvolution_1D` | Convolution | Deconvolution1D | 0.7102 | 7.0 |
| 10 | `MatMul` | Matrix | MatMul | 0.4336 | None |
