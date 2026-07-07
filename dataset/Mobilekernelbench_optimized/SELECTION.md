# Mobilekernelbench_optimized — 30 ops for OptimizeAgent

Selected from the 190-op set to run through OptimizeAgent. Priorities:

1. **Slower than ncnn native** (most optimization headroom) — 17/30 have speedup_fair<1.0 on arm.

2. **Hard operators** — conv variants (Dense/Group/Winograd/Strassen/3D), Deconv, GEMM/MatMul, Det, TopK, GridSample, Einsum, CumSum, Scatter.

3. **Category diversity** — Tensor×8, Convolution×7, Matrix×5, Normalization×2, Activation×2, Unary×2, Reduction×1, Pooling×1, Binary×1, Trigonometry×1.


> DECOMPOSED ops (LogSoftmax, Gemm_alpha, GlobalPool, Reduce* …) are **deliberately excluded**: pnnx runs them as a native chain, so a QD winner for the monolithic Cand never lands (see audit_decomposed_ops.py).


`speedup_fair` = native_fair_ms / ours_fair_ms (both fp32); <1.0 ⇒ we're slower.


| # | op | category | ncnn layer | speedup_fair | speedup_shipped | why |
|---|----|----------|-----------|-------------:|----------------:|-----|
| 1 | `Dense_Convolution_2D` | Convolution | Convolution | 0.010 | 0.006 | slow+hard: im2col/GEMM conv, biggest gap |
| 2 | `Group_Convolution_2D` | Convolution | ConvolutionDepthWise | 0.036 | 0.048 | slow+hard: grouped conv |
| 3 | `Gemm_no_bias` | Matrix | Gemm | 0.047 | 0.028 | slow+hard: GEMM |
| 4 | `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | Deconvolution | 0.080 | 0.086 | slow+hard: deconv, dilated/padded/strided |
| 5 | `Strassen_Convolution_2D` | Convolution | Convolution | 0.090 | 0.088 | slow+hard: Strassen algorithm |
| 6 | `Winograd_Convolution_2D` | Convolution | Convolution | 0.111 | 0.057 | slow+hard: Winograd algorithm |
| 7 | `ConvTranspose_dilations` | Convolution | Deconvolution | 0.234 | 0.244 | slow: transposed conv w/ dilation |
| 8 | `Sub` | Binary | BinaryOp | 0.250 | 6.250 | slow: binary elementwise |
| 9 | `Softmax` | Activation | Softmax | 0.263 | 0.229 | slow+hard: softmax normalization |
| 10 | `Celu` | Activation | CELU | 0.267 | 0.439 | slow: CELU activation |
| 11 | `MaxPool_2d_ceil` | Pooling | Pooling | 0.333 | 1.333 | slow: pooling ceil-mode |
| 12 | `Concat` | Tensor | Concat | 0.333 | 3.333 | slow: concat/layout |
| 13 | `StridedSlice` | Tensor | Crop | 0.333 | 3.667 | slow+diverse: strided slice/crop |
| 14 | `Einsum_sum_all` | Matrix | Reduction | 0.500 | 0.500 | slow: einsum full reduction |
| 15 | `Floor` | Unary | UnaryOp | 0.676 | 2.206 | slow: unary rounding |
| 16 | `Exp` | Unary | UnaryOp | 0.800 | 1.277 | slow: unary transcendental |
| 17 | `Cos` | Trigonometry | UnaryOp | 0.835 | 0.651 | slow: unary trig |
| 18 | `CumSum` | Tensor | CumulativeSum | 1.000 | 1.000 | hard+diverse: prefix scan |
| 19 | `InstanceNormalization` | Normalization | InstanceNorm | 1.000 | 1.373 | hard+diverse: instance norm |
| 20 | `DepthToSpace` | Tensor | PixelShuffle | 1.357 | 1.053 | diverse: pixel-shuffle layout |
| 21 | `LayerNorm` | Normalization | LayerNorm | 1.771 | 1.610 | hard+diverse: transformer LayerNorm |
| 22 | `Conv3D` | Convolution | Convolution3D | — | — | hard+diverse: 3D convolution |
| 23 | `MatMul` | Matrix | MatMul | — | — | hard+diverse: batched matmul |
| 24 | `Einsum` | Matrix | Einsum | — | — | hard+diverse: general einsum |
| 25 | `TopK` | Tensor | torch.topk | — | — | hard+diverse: sort/select |
| 26 | `ArgMax` | Reduction | torch.argmax | — | — | hard+diverse: arg-reduction |
| 27 | `GridSample` | Tensor | GridSample | — | — | hard+diverse: grid sampling/interp |
| 28 | `ScatterElements` | Tensor | aten::scatter | — | — | hard+diverse: scatter/indexing |
| 29 | `Gather` | Tensor | torch.index_select | — | — | hard+diverse: gather/index_select |
| 30 | `Det` | Matrix | aten::linalg_det | — | — | hard+diverse: matrix determinant |
