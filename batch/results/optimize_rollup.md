# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.199×** mean 1.599× max 6.646× · improved(>1.02×) 24/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


| op | cat | regime | rounds | cov | best_rnd | baseline_ms | best_ms | self_speedup | flag | stopped |
|----|-----|--------|-------:|----:|---------:|------------:|--------:|-------------:|------|---------|
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 18 | 3 | 0 | 0.376 | 0.057 | 6.646 | real | budget (15) reached |
| `Conv` | Convolution | compute_bound | 15 | 3 | 0 | 0.141 | 0.047 | 3.015 | real | budget (15) reached |
| `Dense_Convolution_2D` | Convolution | compute_bound | 16 | 3 | 0 | 51.597 | 17.753 | 2.906 | real | budget (15) reached |
| `CumSum` | Tensor | memory_bound | 15 | 2 | 0 | 40.901 | 17.618 | 2.322 | real | budget (15) reached |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 16 | 3 | 0 | 1.463 | 0.690 | 2.122 | real | budget (15) reached |
| `Conv_with_strides_padding` | Convolution | compute_bound | 15 | 3 | 0 | 0.194 | 0.095 | 2.052 | real | budget (15) reached |
| `Conv3D` | Convolution | compute_bound | 16 | 3 | 0 | 45.377 | 25.920 | 1.751 | real | budget (15) reached |
| `LayerNorm` | Normalization | memory_bound | 17 | 3 | 0 | 0.056 | 0.035 | 1.618 | real | budget (15) reached |
| `DepthToSpace` | Tensor | memory_bound | 16 | 2 | 0 | 0.026 | 0.017 | 1.557 | real | budget (15) reached |
| `ConvTranspose_dilations` | Convolution | compute_bound | 19 | 3 | 0 | 27.213 | 18.235 | 1.492 | real | budget (15) reached |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 16 | 3 | 0 | 27.341 | 19.747 | 1.385 | real | budget (15) reached |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 15 | 2 | 0 | 0.952 | 0.712 | 1.338 | real | budget (15) reached |
| `Gemm_no_bias` | Matrix | compute_bound | 18 | 2 | 0 | 20.229 | 15.614 | 1.296 | real | budget (15) reached |
| `InstanceNormalization` | Normalization | memory_bound | 18 | 3 | 0 | 0.066 | 0.052 | 1.285 | real | budget (15) reached |
| `Group_Convolution_2D` | Convolution | compute_bound | 19 | 3 | 0 | 20.739 | 16.831 | 1.232 | real | budget (15) reached |
| `Softplus_3d` | Activation | memory_bound | 15 | 2 | 0 | 18.415 | 15.793 | 1.166 | real | budget (15) reached |
| `MatMul` | Matrix | compute_bound | 6 | 3 | 0 | 18.737 | 16.163 | 1.159 | real | budget (6) reached |
| `Cos` | Trigonometry | memory_bound | 18 | 3 | 0 | 0.106 | 0.093 | 1.144 | real | budget (15) reached |
| `Einsum_sum_all` | Matrix | compute_bound | 15 | 4 | 0 | 18.157 | 15.939 | 1.139 | real | budget (15) reached |
| `Exp` | Unary | memory_bound | 18 | 5 | 0 | 0.194 | 0.177 | 1.095 | real | budget (15) reached |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 15 | 3 | 0 | 21.347 | 19.746 | 1.081 | real | budget (15) reached |
| `BatchNormalization` | Normalization | memory_bound | 15 | 3 | 0 | 0.126 | 0.119 | 1.053 | real | budget (15) reached |
| `Sinh` | Trigonometry | memory_bound | 15 | 3 | 0 | 5.646 | 5.386 | 1.048 | real | budget (15) reached |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 18 | 2 | 0 | 17.742 | 17.194 | 1.032 | real | budget (15) reached |
| `Tan` | Trigonometry | memory_bound | 18 | 3 | 0 | 1.017 | 0.998 | 1.019 | real | budget (15) reached |
| `Floor` | Unary | memory_bound | 15 | 3 | 0 | 0.134 | 0.132 | 1.010 | real | budget (15) reached |
| `Round` | Unary | memory_bound | 15 | 3 | 0 | 3.555 | 3.531 | 1.007 | real | budget (15) reached |
| `Clip` | Tensor | memory_bound | 18 | 4 | 0 | 6.705 | 6.692 | 1.002 | real | budget (15) reached |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 16 | 3 | -1 | 1.167 | 1.167 | 1.000 | real | converged (patience) |
| `ReduceMean` | Reduction | memory_bound | 17 | 4 | -1 | 1.136 | 1.136 | 1.000 | real | converged (patience) |
