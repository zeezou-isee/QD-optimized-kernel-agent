# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.199×** mean 1.599× max 6.646× · improved(>1.02×) 24/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


`rounds` = QD candidates explored · `kept` = rounds that set a new best (the actual optimization steps). A win means best_kernel is a *different* LLM-varied + param-tuned kernel that measured faster on the phone.


`bin` = BD niche (axis1/axis2). `base_bin`→`win_bin` shows which niche the baseline sat in and which niche produced the fastest kernel.


| op | cat | regime | rounds | kept | cov | base_bin | win_bin | baseline_ms | best_ms | self_speedup | flag |
|----|-----|--------|-------:|-----:|----:|----------|---------|------------:|--------:|-------------:|------|
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 18 | 2 | 3 | direct/scalar | gemm/vec | 0.376 | 0.057 | 6.646 | real |
| `Conv` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | direct/vec | 0.141 | 0.047 | 3.015 | real |
| `Dense_Convolution_2D` | Convolution | compute_bound | 16 | 4 | 3 | direct/scalar | direct/vec | 51.597 | 17.753 | 2.906 | real |
| `CumSum` | Tensor | memory_bound | 15 | 1 | 2 | nchw/none | packed/single | 40.901 | 17.618 | 2.322 | real |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 16 | 3 | 3 | direct/scalar | direct/vec | 1.463 | 0.690 | 2.122 | real |
| `Conv_with_strides_padding` | Convolution | compute_bound | 15 | 2 | 3 | direct/scalar | gemm/vec | 0.194 | 0.095 | 2.052 | real |
| `Conv3D` | Convolution | compute_bound | 16 | 2 | 3 | direct/scalar | direct/vec | 45.377 | 25.920 | 1.751 | real |
| `LayerNorm` | Normalization | memory_bound | 17 | 3 | 3 | nchw/none | nhwc/single | 0.056 | 0.035 | 1.618 | real |
| `DepthToSpace` | Tensor | memory_bound | 16 | 1 | 2 | nchw/none | nchw/single | 0.026 | 0.017 | 1.557 | real |
| `ConvTranspose_dilations` | Convolution | compute_bound | 19 | 4 | 3 | direct/scalar | direct/vec | 27.213 | 18.235 | 1.492 | real |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 16 | 3 | 3 | direct/scalar | winograd/vec | 27.341 | 19.747 | 1.385 | real |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 15 | 2 | 2 | nchw/none | nchw/single | 0.952 | 0.712 | 1.338 | real |
| `Gemm_no_bias` | Matrix | compute_bound | 18 | 3 | 2 | direct/scalar | gemm/vec | 20.229 | 15.614 | 1.296 | real |
| `InstanceNormalization` | Normalization | memory_bound | 18 | 2 | 3 | nchw/none | nchw/single | 0.066 | 0.052 | 1.285 | real |
| `Group_Convolution_2D` | Convolution | compute_bound | 19 | 5 | 3 | direct/scalar | direct/vec | 20.739 | 16.831 | 1.232 | real |
| `Softplus_3d` | Activation | memory_bound | 15 | 1 | 2 | nchw/none | nchw/single | 18.415 | 15.793 | 1.166 | real |
| `MatMul` | Matrix | compute_bound | 6 | 2 | 3 | direct/scalar | direct/vec | 18.737 | 16.163 | 1.159 | real |
| `Cos` | Trigonometry | memory_bound | 18 | 4 | 3 | nchw/none | nchw/none | 0.106 | 0.093 | 1.144 | real |
| `Einsum_sum_all` | Matrix | compute_bound | 15 | 3 | 4 | direct/scalar | direct/vec | 18.157 | 15.939 | 1.139 | real |
| `Exp` | Unary | memory_bound | 18 | 4 | 5 | nchw/none | nchw/single | 0.194 | 0.177 | 1.095 | real |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | gemm/vec | 21.347 | 19.746 | 1.081 | real |
| `BatchNormalization` | Normalization | memory_bound | 15 | 3 | 3 | nchw/none | nchw/none | 0.126 | 0.119 | 1.053 | real |
| `Sinh` | Trigonometry | memory_bound | 15 | 2 | 3 | nchw/none | nchw/single | 5.646 | 5.386 | 1.048 | real |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 18 | 2 | 2 | direct/scalar | gemm/vec | 17.742 | 17.194 | 1.032 | real |
| `Tan` | Trigonometry | memory_bound | 18 | 2 | 3 | nchw/none | nhwc/single | 1.017 | 0.998 | 1.019 | real |
| `Floor` | Unary | memory_bound | 15 | 3 | 3 | nchw/none | nchw/none | 0.134 | 0.132 | 1.010 | real |
| `Round` | Unary | memory_bound | 15 | 2 | 3 | nchw/none | nchw/single | 3.555 | 3.531 | 1.007 | real |
| `Clip` | Tensor | memory_bound | 18 | 4 | 4 | nchw/none | nhwc/none | 6.705 | 6.692 | 1.002 | real |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 16 | 2 | 3 | nchw/none | nchw/none | 1.167 | 1.167 | 1.000 | real |
| `ReduceMean` | Reduction | memory_bound | 17 | 4 | 4 | nchw/none | nchw/none | 1.136 | 1.136 | 1.000 | real |

## Covered bins per op (niche → best latency ms; ⚑=winner, ○=baseline niche)

- **Group_Convolution_2D_kernel** (3 bins): `gemm/vec`=0.0565 ⚑ · `direct/vec`=0.1698 · `direct/scalar`=0.3755 ○
- **Conv** (3 bins): `direct/vec`=0.0466 ⚑ · `direct/scalar`=0.1405 ○ · `gemm/vec`=0.1669
- **Dense_Convolution_2D** (3 bins): `direct/vec`=17.75 ⚑ · `gemm/vec`=28.73 · `direct/scalar`=51.6 ○
- **CumSum** (2 bins): `packed/single`=17.62 ⚑ · `nchw/none`=40.9 ○
- **Winograd_Convolution_2D_padding** (3 bins): `direct/vec`=0.6895 ⚑ · `gemm/vec`=0.7087 · `direct/scalar`=1.463 ○
- **Conv_with_strides_padding** (3 bins): `gemm/vec`=0.0945 ⚑ · `direct/vec`=0.1091 · `direct/scalar`=0.1939 ○
- **Conv3D** (3 bins): `direct/vec`=25.92 ⚑ · `gemm/vec`=27.22 · `direct/scalar`=45.38 ○
- **LayerNorm** (3 bins): `nhwc/single`=0.0348 ⚑ · `nchw/single`=0.0352 · `nchw/none`=0.0399 ○
- **DepthToSpace** (2 bins): `nchw/single`=0.0167 ⚑ · `nchw/none`=0.026 ○
- **ConvTranspose_dilations** (3 bins): `direct/vec`=18.24 ⚑ · `direct/scalar`=27.21 ○ · `gemm/vec`=28.64
- **Winograd_Convolution_2D** (3 bins): `winograd/vec`=19.75 ⚑ · `gemm/vec`=20.58 · `direct/scalar`=27.34 ○
- **DeconvolutionDepthwise_2D_stride** (2 bins): `nchw/single`=0.7116 ⚑ · `nchw/none`=0.9522 ○
- **Gemm_no_bias** (2 bins): `gemm/vec`=15.61 ⚑ · `direct/scalar`=20.23 ○
- **InstanceNormalization** (3 bins): `nchw/single`=0.0516 ⚑ · `nhwc/single`=0.0594 · `nchw/none`=0.0663 ○
- **Group_Convolution_2D** (3 bins): `direct/vec`=16.83 ⚑ · `gemm/vec`=16.9 · `direct/scalar`=20.74 ○
- **Softplus_3d** (2 bins): `nchw/single`=15.79 ⚑ · `nchw/none`=18.42 ○
- **MatMul** (3 bins): `direct/vec`=16.16 ⚑ · `gemm/vec`=16.85 · `direct/scalar`=18.74 ○
- **Cos** (3 bins): `nchw/none`=0.0926 ⚑ · `nchw/single`=0.0951 · `packed/single`=0.1045
- **Einsum_sum_all** (4 bins): `direct/vec`=15.94 ⚑ · `tree/vec`=16.38 · `direct/scalar`=18.16 ○ · `gemm/dotprod`=19.07
- **Exp** (5 bins): `nchw/single`=0.1768 ⚑ · `nhwc/none`=0.1787 · `packed/double`=0.1917 · `packed/single`=0.1927 · `nchw/none`=0.1936 ○
- **Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__** (3 bins): `gemm/vec`=19.75 ⚑ · `direct/scalar`=21.35 ○ · `gemm/scalar`=24.07
- **BatchNormalization** (3 bins): `nchw/none`=0.1193 ⚑ · `nchw/single`=0.1201 · `nhwc/single`=0.3091
- **Sinh** (3 bins): `nchw/single`=5.386 ⚑ · `nchw/none`=5.646 ○ · `nhwc/single`=15.17
- **Strassen_Convolution_2D** (2 bins): `gemm/vec`=17.19 ⚑ · `direct/scalar`=17.74 ○
- **Tan** (3 bins): `nhwc/single`=0.9978 ⚑ · `nchw/single`=1 · `nchw/none`=1.017 ○
- **Floor** (3 bins): `nchw/none`=0.1323 ⚑ · `packed/none`=0.1323 · `nhwc/single`=0.1327
- **Round** (3 bins): `nchw/single`=3.531 ⚑ · `nchw/none`=3.555 ○ · `nhwc/single`=3.559
- **Clip** (4 bins): `nhwc/none`=6.692 ⚑ · `nchw/none`=6.705 ○ · `nhwc/single`=6.715 · `nchw/single`=6.727
- **ReduceMax_with_negative_values** (3 bins): `nchw/none`=1.167 ⚑ · `nhwc/double`=3.038 · `nhwc/single`=3.737
- **ReduceMean** (4 bins): `nchw/none`=1.136 ⚑ · `nchw/single`=2.955 · `nchw/double`=2.962 · `packed/single`=3.025
