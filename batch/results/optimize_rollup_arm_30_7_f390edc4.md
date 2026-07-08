# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.543×** mean 2.816× max 7.790× · improved(>1.02×) 26/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


`rounds` = QD candidates explored · `kept` = rounds that set a new best (the actual optimization steps). A win means best_kernel is a *different* LLM-varied + param-tuned kernel that measured faster on the phone.


`bin` = BD niche (axis1/axis2). `base_bin`→`win_bin` shows which niche the baseline sat in and which niche produced the fastest kernel.


| op | cat | regime | rounds | kept | cov | base_bin | win_bin | baseline_ms | best_ms | self_speedup | flag |
|----|-----|--------|-------:|-----:|----:|----------|---------|------------:|--------:|-------------:|------|
| `Dense_Convolution_1D` | Convolution | compute_bound | 27 | 3 | 3 | direct/scalar | gemm/vec | 2.233 | 0.287 | 7.790 | real |
| `Dense_Convolution_2D` | Convolution | compute_bound | 35 | 4 | 3 | direct/scalar | gemm/vec | 9.739 | 1.272 | 7.656 | real |
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 29 | 2 | 3 | direct/scalar | gemm/vec | 0.266 | 0.035 | 7.484 | real |
| `Conv3D` | Convolution | compute_bound | 24 | 4 | 3 | direct/scalar | gemm/vec | 39.102 | 5.886 | 6.643 | real |
| `Dense_Convolution_2D_kernel` | Convolution | compute_bound | 30 | 3 | 3 | direct/scalar | direct/vec | 224.729 | 37.313 | 6.023 | real |
| `Atanh` | Trigonometry | memory_bound | 32 | 4 | 4 | nchw/none | nchw/single | 0.206 | 0.035 | 5.903 | real |
| `DeconvolutionDepthwise_2D_kernel` | Convolution | memory_bound | 16 | 2 | 3 | nchw/none | nhwc/single | 1.229 | 0.243 | 5.058 | real |
| `Group_Convolution_2D` | Convolution | compute_bound | 15 | 4 | 3 | direct/scalar | gemm/vec | 2.188 | 0.502 | 4.355 | real |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 36 | 3 | 3 | direct/scalar | winograd/vec | 22.752 | 7.377 | 3.084 | real |
| `Conv` | Convolution | compute_bound | 32 | 3 | 3 | direct/scalar | direct/vec | 0.180 | 0.059 | 3.022 | real |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 31 | 5 | 3 | direct/scalar | gemm/vec | 0.832 | 0.328 | 2.534 | real |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 23 | 3 | 3 | direct/scalar | gemm/vec | 1.464 | 0.583 | 2.509 | real |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 26 | 3 | 3 | nchw/none | nchw/none | 1.161 | 0.637 | 1.822 | real |
| `ReduceMean` | Reduction | memory_bound | 30 | 3 | 2 | nchw/none | nchw/none | 1.003 | 0.606 | 1.656 | real |
| `Asinh` | Trigonometry | memory_bound | 34 | 4 | 4 | nchw/none | nchw/single | 0.391 | 0.247 | 1.582 | real |
| `Asin` | Trigonometry | memory_bound | 36 | 6 | 3 | nchw/none | packed/none | 4.354 | 2.893 | 1.505 | real |
| `MatMul` | Matrix | compute_bound | 35 | 4 | 4 | direct/scalar | gemm/vec | 0.432 | 0.307 | 1.404 | real |
| `Sinh` | Trigonometry | memory_bound | 30 | 4 | 4 | nchw/none | nchw/single | 5.590 | 4.358 | 1.283 | real |
| `Cos` | Trigonometry | memory_bound | 30 | 5 | 4 | nchw/none | nchw/single | 2.048 | 1.597 | 1.283 | real |
| `Exp` | Unary | memory_bound | 35 | 5 | 4 | nchw/none | packed/double | 2.878 | 2.311 | 1.245 | real |
| `Cosh` | Trigonometry | memory_bound | 32 | 3 | 3 | nchw/none | packed/single | 0.241 | 0.206 | 1.165 | real |
| `Clip` | Tensor | memory_bound | 33 | 4 | 3 | nchw/none | nhwc/single | 6.074 | 5.278 | 1.151 | real |
| `Tan` | Trigonometry | memory_bound | 34 | 4 | 5 | nchw/none | packed/single | 1.010 | 0.881 | 1.147 | real |
| `Round` | Unary | memory_bound | 29 | 5 | 5 | nchw/none | nhwc/none | 3.042 | 2.791 | 1.090 | real |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 31 | 3 | 3 | direct/scalar | gemm/vec | 3.645 | 3.450 | 1.057 | real |
| `Floor` | Unary | memory_bound | 30 | 2 | 3 | nchw/none | nhwc/single | 1.242 | 1.190 | 1.044 | real |
| `ConvTranspose_dilations` | Convolution | compute_bound | 14 | 2 | 3 | direct/scalar | direct/scalar | 13.849 | 13.849 | 1.000 | real |
| `Deconvolution_1D` | Convolution | compute_bound | 19 | 2 | 3 | direct/scalar | direct/scalar | 0.712 | 0.712 | 1.000 | real |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 16 | 2 | 2 | nchw/none | nchw/none | 0.838 | 0.838 | 1.000 | real |
| `ReduceMin_with_negative_values` | Reduction | memory_bound | 27 | 1 | 2 | nchw/none | nchw/none | 1.245 | 1.245 | 1.000 | real |

## Covered bins per op (niche → best latency ms; ⚑=winner, ○=baseline niche)

- **Dense_Convolution_1D** (3 bins): `gemm/vec`=0.2867 ⚑ · `direct/vec`=0.3182 · `direct/scalar`=1.155 ○
- **Dense_Convolution_2D** (3 bins): `gemm/vec`=1.272 ⚑ · `direct/vec`=1.375 · `direct/scalar`=9.739 ○
- **Group_Convolution_2D_kernel** (3 bins): `gemm/vec`=0.0355 ⚑ · `direct/vec`=0.0391 · `direct/scalar`=0.2657 ○
- **Conv3D** (3 bins): `gemm/vec`=5.886 ⚑ · `direct/scalar`=17.4 ○ · `direct/vec`=17.8
- **Dense_Convolution_2D_kernel** (3 bins): `direct/vec`=37.31 ⚑ · `gemm/vec`=61.48 · `direct/scalar`=224.7 ○
- **Atanh** (4 bins): `nchw/single`=0.0349 ⚑ · `nhwc/none`=0.1879 · `nhwc/single`=0.1897 · `nchw/none`=0.206 ○
- **DeconvolutionDepthwise_2D_kernel** (3 bins): `nhwc/single`=0.243 ⚑ · `nchw/single`=0.2589 · `nchw/none`=1.229 ○
- **Group_Convolution_2D** (3 bins): `gemm/vec`=0.5025 ⚑ · `direct/vec`=0.5174 · `direct/scalar`=0.5401 ○
- **Winograd_Convolution_2D** (3 bins): `winograd/vec`=7.377 ⚑ · `gemm/vec`=16.52 · `direct/scalar`=22.75 ○
- **Conv** (3 bins): `direct/vec`=0.0595 ⚑ · `gemm/vec`=0.1544 · `direct/scalar`=0.1798 ○
- **Strassen_Convolution_2D** (3 bins): `gemm/vec`=0.3284 ⚑ · `direct/vec`=0.4981 · `direct/scalar`=0.832 ○
- **Winograd_Convolution_2D_padding** (3 bins): `gemm/vec`=0.5832 ⚑ · `direct/vec`=1.343 · `direct/scalar`=1.393 ○
- **DeconvolutionDepthwise_2D_stride** (3 bins): `nchw/none`=0.637 ⚑ · `nchw/single`=3.08 · `nhwc/single`=3.426
- **ReduceMean** (2 bins): `nchw/none`=0.6055 ⚑ · `nchw/single`=0.9091
- **Asinh** (4 bins): `nchw/single`=0.2472 ⚑ · `nhwc/none`=0.3099 · `packed/single`=0.3856 · `nchw/none`=0.3911 ○
- **Asin** (3 bins): `packed/none`=2.893 ⚑ · `nchw/single`=2.966 · `nchw/none`=3.732 ○
- **MatMul** (4 bins): `gemm/vec`=0.3075 ⚑ · `direct/vec`=0.3773 · `direct/scalar`=0.4318 ○ · `gemm/dotprod`=1.847
- **Sinh** (4 bins): `nchw/single`=4.358 ⚑ · `packed/double`=4.6 · `nchw/none`=5.59 ○ · `nhwc/none`=12.43
- **Cos** (4 bins): `nchw/single`=1.597 ⚑ · `nhwc/none`=1.657 · `packed/single`=1.671 · `nchw/none`=2.048 ○
- **Exp** (4 bins): `packed/double`=2.311 ⚑ · `nhwc/single`=2.47 · `packed/single`=2.778 · `nchw/none`=2.878 ○
- **Cosh** (3 bins): `packed/single`=0.2065 ⚑ · `nchw/none`=0.2406 ○ · `nhwc/single`=0.2431
- **Clip** (3 bins): `nhwc/single`=5.278 ⚑ · `nchw/none`=5.919 ○ · `packed/none`=6.074
- **Tan** (5 bins): `packed/single`=0.8807 ⚑ · `nchw/none`=1.01 ○ · `packed/double`=1.013 · `nchw/single`=1.018 · `nhwc/single`=1.024
- **Round** (5 bins): `nhwc/none`=2.791 ⚑ · `nchw/single`=2.925 · `packed/single`=2.934 · `nchw/double`=3.028 · `nchw/none`=3.042 ○
- **Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__** (3 bins): `gemm/vec`=3.45 ⚑ · `direct/scalar`=3.645 ○ · `direct/vec`=14.36
- **Floor** (3 bins): `nhwc/single`=1.19 ⚑ · `nchw/single`=1.193 · `nchw/none`=1.242 ○
- **ConvTranspose_dilations** (3 bins): `direct/scalar`=13.85 ⚑ · `gemm/vec`=30.73 · `gemm/scalar`=93.21
- **Deconvolution_1D** (3 bins): `direct/scalar`=0.7116 ⚑ · `gemm/vec`=2.53 · `gemm/scalar`=2.566
- **ReduceMax_with_negative_values** (2 bins): `nchw/none`=0.8379 ⚑ · `nchw/single`=1.999
- **ReduceMin_with_negative_values** (2 bins): `nchw/none`=1.245 ⚑ · `nchw/single`=2.433
