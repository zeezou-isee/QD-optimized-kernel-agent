# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.301×** mean 2.337× max 7.950× · improved(>1.02×) 23/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


`rounds` = QD candidates explored · `kept` = rounds that set a new best (the actual optimization steps). A win means best_kernel is a *different* LLM-varied + param-tuned kernel that measured faster on the phone.


`bin` = BD niche (axis1/axis2). `base_bin`→`win_bin` shows which niche the baseline sat in and which niche produced the fastest kernel.


| op | cat | regime | rounds | kept | cov | base_bin | win_bin | baseline_ms | best_ms | self_speedup | flag |
|----|-----|--------|-------:|-----:|----:|----------|---------|------------:|--------:|-------------:|------|
| `Dense_Convolution_1D` | Convolution | compute_bound | 15 | 2 | 3 | direct/scalar | gemm/vec | 2.231 | 0.281 | 7.950 | real |
| `Dense_Convolution_2D` | Convolution | compute_bound | 17 | 4 | 3 | direct/scalar | direct/vec | 9.438 | 1.200 | 7.862 | real |
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 17 | 3 | 3 | direct/scalar | gemm/vec | 0.265 | 0.036 | 7.438 | real |
| `Conv` | Convolution | compute_bound | 15 | 2 | 3 | direct/scalar | direct/vec | 0.175 | 0.034 | 5.114 | real |
| `Dense_Convolution_2D_kernel` | Convolution | compute_bound | 19 | 3 | 3 | direct/scalar | direct/vec | 224.359 | 48.460 | 4.630 | real |
| `Group_Convolution_2D` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | gemm/vec | 2.235 | 0.542 | 4.125 | real |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | gemm/vec | 1.462 | 0.546 | 2.678 | real |
| `Conv3D` | Convolution | compute_bound | 18 | 3 | 3 | direct/scalar | direct/vec | 38.811 | 16.230 | 2.391 | real |
| `Cosh` | Trigonometry | memory_bound | 15 | 3 | 4 | nchw/none | nhwc/none | 0.243 | 0.107 | 2.261 | real |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 15 | 3 | 3 | direct/scalar | direct/vec | 1.158 | 0.578 | 2.003 | real |
| `MatMul` | Matrix | compute_bound | 15 | 2 | 2 | direct/scalar | gemm/vec | 0.434 | 0.277 | 1.568 | real |
| `ReduceMean` | Reduction | memory_bound | 15 | 3 | 2 | nchw/none | nchw/none | 1.036 | 0.689 | 1.504 | real |
| `Sinh` | Trigonometry | memory_bound | 15 | 3 | 3 | nchw/none | nchw/none | 5.617 | 3.801 | 1.478 | real |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 17 | 3 | 3 | nchw/none | nchw/single | 1.150 | 0.819 | 1.404 | real |
| `Asin` | Trigonometry | memory_bound | 16 | 3 | 3 | nchw/none | nhwc/none | 4.085 | 3.126 | 1.307 | real |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 19 | 4 | 4 | direct/scalar | winograd/vec | 25.017 | 19.324 | 1.295 | real |
| `Asinh` | Trigonometry | memory_bound | 15 | 3 | 4 | nchw/none | nhwc/single | 0.392 | 0.307 | 1.276 | real |
| `Floor` | Unary | memory_bound | 18 | 4 | 5 | nchw/none | packed/single | 1.335 | 1.112 | 1.200 | real |
| `Tan` | Trigonometry | memory_bound | 15 | 3 | 4 | nchw/none | nhwc/none | 1.026 | 0.859 | 1.194 | real |
| `Clip` | Tensor | memory_bound | 15 | 2 | 2 | nchw/none | nhwc/single | 7.119 | 6.131 | 1.161 | real |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 15 | 2 | 3 | direct/scalar | gemm/vec | 3.911 | 3.429 | 1.140 | real |
| `Round` | Unary | memory_bound | 15 | 3 | 3 | nchw/none | nhwc/single | 3.197 | 2.993 | 1.068 | real |
| `Cos` | Trigonometry | memory_bound | 19 | 4 | 5 | nchw/none | packed/single | 1.760 | 1.662 | 1.059 | real |
| `Exp` | Unary | memory_bound | 15 | 2 | 3 | nchw/none | packed/none | 2.828 | 2.789 | 1.014 | real |
| `ConvTranspose_dilations` | Convolution | compute_bound | 11 | 2 | 2 | direct/scalar | direct/scalar | 13.940 | 13.940 | 1.000 | real |
| `Atanh` | Trigonometry | memory_bound | 15 | 2 | 2 | nchw/none | nchw/none | 0.150 | 0.150 | 1.000 | real |
| `DeconvolutionDepthwise_2D_kernel` | Convolution | memory_bound | 16 | 1 | 2 | nchw/none | nchw/none | 1.456 | 1.456 | 1.000 | real |
| `Deconvolution_1D` | Convolution | compute_bound | 18 | 3 | 3 | direct/scalar | direct/scalar | 0.685 | 0.685 | 1.000 | real |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 14 | 1 | 2 | nchw/none | nchw/none | 1.019 | 1.019 | 1.000 | real |
| `ReduceMin_with_negative_values` | Reduction | memory_bound | 16 | 0 | 1 | nchw/none | nchw/none | 1.280 | 1.280 | 1.000 | real |

## Covered bins per op (niche → best latency ms; ⚑=winner, ○=baseline niche)

- **Dense_Convolution_1D** (3 bins): `gemm/vec`=0.2806 ⚑ · `direct/vec`=0.3434 · `direct/scalar`=2.231 ○
- **Dense_Convolution_2D** (3 bins): `direct/vec`=1.2 ⚑ · `gemm/vec`=1.857 · `direct/scalar`=9.438 ○
- **Group_Convolution_2D_kernel** (3 bins): `gemm/vec`=0.0356 ⚑ · `direct/vec`=0.0381 · `direct/scalar`=0.2648 ○
- **Conv** (3 bins): `direct/vec`=0.0343 ⚑ · `gemm/vec`=0.1479 · `direct/scalar`=0.1754 ○
- **Dense_Convolution_2D_kernel** (3 bins): `direct/vec`=48.46 ⚑ · `gemm/vec`=132.4 · `direct/scalar`=224.4 ○
- **Group_Convolution_2D** (3 bins): `gemm/vec`=0.5417 ⚑ · `direct/vec`=1.097 · `direct/scalar`=2.235 ○
- **Winograd_Convolution_2D_padding** (3 bins): `gemm/vec`=0.546 ⚑ · `direct/scalar`=0.7663 ○ · `direct/vec`=1.429
- **Conv3D** (3 bins): `direct/vec`=16.23 ⚑ · `gemm/vec`=16.75 · `direct/scalar`=38.81 ○
- **Cosh** (4 bins): `nhwc/none`=0.1075 ⚑ · `nchw/none`=0.2431 ○ · `nchw/single`=0.245 · `nhwc/single`=0.5042
- **Strassen_Convolution_2D** (3 bins): `direct/vec`=0.5779 ⚑ · `direct/scalar`=1.158 ○ · `gemm/vec`=3.095
- **MatMul** (2 bins): `gemm/vec`=0.2765 ⚑ · `direct/scalar`=0.4336 ○
- **ReduceMean** (2 bins): `nchw/none`=0.6885 ⚑ · `nchw/single`=1.167
- **Sinh** (3 bins): `nchw/none`=3.801 ⚑ · `nchw/single`=4.685 · `packed/single`=4.784
- **DeconvolutionDepthwise_2D_stride** (3 bins): `nchw/single`=0.8191 ⚑ · `nhwc/none`=1.106 · `nchw/none`=1.15 ○
- **Asin** (3 bins): `nhwc/none`=3.126 ⚑ · `packed/single`=3.581 · `nchw/none`=4.085 ○
- **Winograd_Convolution_2D** (4 bins): `winograd/vec`=19.32 ⚑ · `gemm/vec`=20.13 · `direct/scalar`=25.02 ○ · `gemm/scalar`=26.96
- **Asinh** (4 bins): `nhwc/single`=0.3068 ⚑ · `packed/double`=0.3077 · `nhwc/none`=0.3084 · `nchw/none`=0.3916 ○
- **Floor** (5 bins): `packed/single`=1.112 ⚑ · `nhwc/none`=1.237 · `nchw/single`=1.244 · `nhwc/single`=1.276 · `nchw/none`=1.335 ○
- **Tan** (4 bins): `nhwc/none`=0.8591 ⚑ · `packed/single`=1.024 · `nchw/none`=1.026 ○ · `nhwc/single`=2.902
- **Clip** (2 bins): `nhwc/single`=6.131 ⚑ · `nchw/none`=6.152 ○
- **Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__** (3 bins): `gemm/vec`=3.429 ⚑ · `direct/scalar`=3.911 ○ · `gemm/scalar`=5.958
- **Round** (3 bins): `nhwc/single`=2.993 ⚑ · `nchw/single`=3.039 · `nchw/none`=3.197 ○
- **Cos** (5 bins): `packed/single`=1.662 ⚑ · `nhwc/none`=1.68 · `packed/double`=1.687 · `nchw/none`=1.76 ○ · `nchw/double`=1.796
- **Exp** (3 bins): `packed/none`=2.789 ⚑ · `nchw/single`=2.823 · `nchw/none`=2.828 ○
- **ConvTranspose_dilations** (2 bins): `direct/scalar`=13.94 ⚑ · `gemm/vec`=30.57
- **Atanh** (2 bins): `nchw/none`=0.1498 ⚑ · `nhwc/single`=0.1929
- **DeconvolutionDepthwise_2D_kernel** (2 bins): `nchw/none`=1.456 ⚑ · `nchw/single`=18.27
- **Deconvolution_1D** (3 bins): `direct/scalar`=0.6852 ⚑ · `gemm/scalar`=2.353 · `gemm/vec`=2.904
- **ReduceMax_with_negative_values** (2 bins): `nchw/none`=1.019 ⚑ · `nchw/single`=2.097
- **ReduceMin_with_negative_values** (1 bins): `nchw/none`=1.28 ⚑
