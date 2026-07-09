# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.482×** mean 2.580× max 7.887× · improved(>1.02×) 27/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


`rounds` = QD candidates explored · `kept` = rounds that set a new best (the actual optimization steps). A win means best_kernel is a *different* LLM-varied + param-tuned kernel that measured faster on the phone.


`bin` = BD niche (axis1/axis2). `base_bin`→`win_bin` shows which niche the baseline sat in and which niche produced the fastest kernel.


| op | cat | regime | rounds | kept | cov | base_bin | win_bin | baseline_ms | best_ms | self_speedup | flag |
|----|-----|--------|-------:|-----:|----:|----------|---------|------------:|--------:|-------------:|------|
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 36 | 5 | 3 | direct/scalar | direct/scalar | 0.266 | 0.034 | 7.887 | real |
| `Dense_Convolution_2D` | Convolution | compute_bound | 27 | 2 | 3 | direct/scalar | gemm/vec | 9.308 | 1.198 | 7.768 | real |
| `Conv3D` | Convolution | compute_bound | 21 | 3 | 3 | direct/scalar | gemm/vec | 39.128 | 5.048 | 7.752 | real |
| `Dense_Convolution_1D` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | gemm/vec | 2.233 | 0.329 | 6.784 | real |
| `Conv` | Convolution | compute_bound | 36 | 2 | 3 | direct/scalar | direct/vec | 0.178 | 0.041 | 4.384 | real |
| `Dense_Convolution_2D_kernel` | Convolution | compute_bound | 50 | 2 | 3 | direct/scalar | direct/vec | 223.728 | 61.717 | 3.625 | real |
| `MatMul` | Matrix | compute_bound | 41 | 3 | 3 | direct/scalar | gemm/vec | 0.433 | 0.144 | 3.007 | real |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 22 | 4 | 3 | direct/scalar | gemm/vec | 0.667 | 0.225 | 2.962 | real |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 49 | 6 | 4 | direct/scalar | winograd/vec | 22.280 | 7.766 | 2.869 | real |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 20 | 2 | 3 | direct/scalar | gemm/vec | 1.394 | 0.496 | 2.808 | real |
| `Group_Convolution_2D` | Convolution | compute_bound | 33 | 3 | 3 | direct/scalar | gemm/vec | 2.131 | 0.779 | 2.735 | real |
| `Cosh` | Trigonometry | memory_bound | 48 | 5 | 5 | nchw/none | packed/none | 0.244 | 0.107 | 2.272 | real |
| `DeconvolutionDepthwise_2D_kernel` | Convolution | memory_bound | 41 | 1 | 2 | nchw/none | nchw/single | 1.232 | 0.594 | 2.074 | real |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 45 | 3 | 3 | nchw/none | nchw/single | 1.156 | 0.703 | 1.643 | real |
| `Deconvolution_1D` | Convolution | compute_bound | 54 | 6 | 4 | direct/scalar | direct/vec | 0.700 | 0.456 | 1.535 | real |
| `Asin` | Trigonometry | memory_bound | 46 | 4 | 4 | nchw/none | nhwc/none | 4.193 | 2.934 | 1.429 | real |
| `Sinh` | Trigonometry | memory_bound | 35 | 5 | 4 | nchw/none | nhwc/none | 5.441 | 4.089 | 1.331 | real |
| `Tan` | Trigonometry | memory_bound | 43 | 2 | 3 | nchw/none | nchw/single | 1.039 | 0.785 | 1.325 | real |
| `Exp` | Unary | memory_bound | 49 | 4 | 4 | nchw/none | nhwc/none | 2.663 | 2.177 | 1.223 | real |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 46 | 7 | 4 | direct/scalar | direct/vec | 3.882 | 3.231 | 1.202 | real |
| `Clip` | Tensor | memory_bound | 46 | 5 | 5 | nchw/none | nchw/single | 5.935 | 5.102 | 1.163 | real |
| `Cos` | Trigonometry | memory_bound | 47 | 5 | 5 | nchw/none | nchw/single | 1.701 | 1.477 | 1.151 | real |
| `Atanh` | Trigonometry | memory_bound | 53 | 5 | 5 | nchw/none | nchw/single | 0.208 | 0.183 | 1.138 | real |
| `Round` | Unary | memory_bound | 50 | 5 | 5 | nchw/none | nhwc/double | 2.869 | 2.613 | 1.098 | real |
| `ReduceMin_with_negative_values` | Reduction | memory_bound | 24 | 3 | 2 | nchw/none | nchw/single | 1.234 | 1.127 | 1.095 | real |
| `Floor` | Unary | memory_bound | 54 | 6 | 4 | nchw/none | nchw/single | 1.241 | 1.145 | 1.084 | real |
| `Asinh` | Trigonometry | memory_bound | 50 | 5 | 4 | nchw/none | nchw/none | 0.391 | 0.374 | 1.044 | real |
| `ConvTranspose_dilations` | Convolution | compute_bound | 13 | 2 | 3 | direct/scalar | direct/scalar | 13.428 | 13.428 | 1.000 | real |
| `ReduceMean` | Reduction | memory_bound | 20 | 4 | 4 | nchw/none | nchw/none | 0.838 | 0.838 | 1.000 | real |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 18 | 1 | 2 | nchw/none | nchw/none | 1.005 | 1.005 | 1.000 | real |

## Covered bins per op (niche → best latency ms; ⚑=winner, ○=baseline niche)

- **Group_Convolution_2D_kernel** (3 bins): `direct/scalar`=0.0337 ⚑ · `gemm/vec`=0.0341 · `direct/vec`=0.0378
- **Dense_Convolution_2D** (3 bins): `gemm/vec`=1.198 ⚑ · `direct/vec`=1.613 · `direct/scalar`=9.308 ○
- **Conv3D** (3 bins): `gemm/vec`=5.048 ⚑ · `direct/scalar`=7.581 ○ · `direct/vec`=11.99
- **Dense_Convolution_1D** (3 bins): `gemm/vec`=0.3292 ⚑ · `direct/vec`=0.6156 · `direct/scalar`=2.233 ○
- **Conv** (3 bins): `direct/vec`=0.0406 ⚑ · `gemm/vec`=0.0432 · `direct/scalar`=0.178 ○
- **Dense_Convolution_2D_kernel** (3 bins): `direct/vec`=61.72 ⚑ · `gemm/vec`=69.49 · `direct/scalar`=223.7 ○
- **MatMul** (3 bins): `gemm/vec`=0.1439 ⚑ · `gemm/dotprod`=0.3922 · `direct/scalar`=0.4327 ○
- **Strassen_Convolution_2D** (3 bins): `gemm/vec`=0.2253 ⚑ · `direct/vec`=0.5239 · `direct/scalar`=0.6674 ○
- **Winograd_Convolution_2D** (4 bins): `winograd/vec`=7.766 ⚑ · `gemm/vec`=13.58 · `direct/scalar`=22.28 ○ · `gemm/scalar`=38.82
- **Winograd_Convolution_2D_padding** (3 bins): `gemm/vec`=0.4962 ⚑ · `direct/scalar`=1.394 ○ · `direct/vec`=1.506
- **Group_Convolution_2D** (3 bins): `gemm/vec`=0.7793 ⚑ · `direct/vec`=0.8913 · `direct/scalar`=2.131 ○
- **Cosh** (5 bins): `packed/none`=0.1073 ⚑ · `nhwc/none`=0.1199 · `packed/single`=0.143 · `nchw/none`=0.2095 ○ · `nchw/single`=0.243
- **DeconvolutionDepthwise_2D_kernel** (2 bins): `nchw/single`=0.5941 ⚑ · `nchw/none`=1.232 ○
- **DeconvolutionDepthwise_2D_stride** (3 bins): `nchw/single`=0.7034 ⚑ · `nhwc/single`=0.868 · `nchw/none`=1.156 ○
- **Deconvolution_1D** (4 bins): `direct/vec`=0.4562 ⚑ · `gemm/vec`=0.6922 · `direct/scalar`=0.7003 ○ · `gemm/scalar`=1.818
- **Asin** (4 bins): `nhwc/none`=2.934 ⚑ · `nchw/single`=3.46 · `nchw/none`=3.761 ○ · `packed/single`=4.181
- **Sinh** (4 bins): `nhwc/none`=4.089 ⚑ · `packed/none`=4.548 · `nchw/single`=4.7 · `nchw/none`=4.978 ○
- **Tan** (3 bins): `nchw/single`=0.7846 ⚑ · `nhwc/none`=1.035 · `nchw/none`=1.039 ○
- **Exp** (4 bins): `nhwc/none`=2.177 ⚑ · `nchw/single`=2.371 · `nhwc/single`=2.627 · `nchw/none`=2.663 ○
- **Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__** (4 bins): `direct/vec`=3.231 ⚑ · `direct/scalar`=3.708 ○ · `gemm/vec`=4.086 · `gemm/scalar`=13.51
- **Clip** (5 bins): `nchw/single`=5.102 ⚑ · `nchw/none`=5.902 ○ · `packed/single`=6.903 · `nhwc/none`=6.971 · `nhwc/single`=10.19
- **Cos** (5 bins): `nchw/single`=1.477 ⚑ · `nchw/none`=1.641 ○ · `nhwc/single`=1.669 · `packed/none`=1.726 · `packed/single`=1.74
- **Atanh** (5 bins): `nchw/single`=0.1826 ⚑ · `nchw/none`=0.1881 ○ · `nhwc/none`=0.1942 · `packed/single`=0.1944 · `nhwc/single`=0.2059
- **Round** (5 bins): `nhwc/double`=2.613 ⚑ · `nchw/single`=2.73 · `nhwc/single`=2.816 · `nchw/none`=2.869 ○ · `nhwc/none`=2.948
- **ReduceMin_with_negative_values** (2 bins): `nchw/single`=1.127 ⚑ · `nchw/none`=1.13 ○
- **Floor** (4 bins): `nchw/single`=1.145 ⚑ · `nhwc/single`=1.154 · `packed/none`=1.226 · `nchw/none`=1.241 ○
- **Asinh** (4 bins): `nchw/none`=0.374 ⚑ · `nchw/single`=0.3868 · `nchw/double`=0.3875 · `nhwc/double`=0.3895
- **ConvTranspose_dilations** (3 bins): `direct/scalar`=13.43 ⚑ · `gemm/scalar`=28.87 · `gemm/vec`=30.56
- **ReduceMean** (4 bins): `nchw/none`=0.8384 ⚑ · `nchw/single`=1.345 · `nchw/double`=1.539 · `nhwc/single`=3.717
- **ReduceMax_with_negative_values** (2 bins): `nchw/none`=1.005 ⚑ · `nchw/single`=3.641
