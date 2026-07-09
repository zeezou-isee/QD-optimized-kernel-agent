# OptimizeAgent rollup — Mobilekernelbench_optimized (arm)

30 ops · real=30 suspect=0 tainted=0 crash=0

real-win self-speedup: median **1.351×** mean 2.396× max 7.979× · improved(>1.02×) 25/30


flags: real=ms-scale trustworthy · suspect=μs below noise floor (0.02ms) · tainted=speedup>8× degenerate/path-mix · crash=no summary (e.g. arm kernel -100)


`rounds` = QD candidates explored · `kept` = rounds that set a new best (the actual optimization steps). A win means best_kernel is a *different* LLM-varied + param-tuned kernel that measured faster on the phone.


`bin` = BD niche (axis1/axis2). `base_bin`→`win_bin` shows which niche the baseline sat in and which niche produced the fastest kernel.


| op | cat | regime | rounds | kept | cov | base_bin | win_bin | baseline_ms | best_ms | self_speedup | flag |
|----|-----|--------|-------:|-----:|----:|----------|---------|------------:|--------:|-------------:|------|
| `Group_Convolution_2D_kernel` | Convolution | compute_bound | 20 | 2 | 3 | direct/scalar | direct/vec | 0.264 | 0.033 | 7.979 | real |
| `Dense_Convolution_1D` | Convolution | compute_bound | 19 | 4 | 3 | direct/scalar | gemm/vec | 2.235 | 0.293 | 7.635 | real |
| `Dense_Convolution_2D` | Convolution | compute_bound | 19 | 5 | 3 | direct/scalar | gemm/vec | 9.729 | 1.292 | 7.531 | real |
| `Conv` | Convolution | compute_bound | 21 | 4 | 3 | direct/scalar | direct/vec | 0.175 | 0.028 | 6.313 | real |
| `Group_Convolution_2D` | Convolution | compute_bound | 21 | 3 | 3 | direct/scalar | direct/vec | 2.171 | 0.497 | 4.369 | real |
| `Strassen_Convolution_2D` | Convolution | compute_bound | 20 | 3 | 3 | direct/scalar | gemm/vec | 1.159 | 0.317 | 3.657 | real |
| `Conv3D` | Convolution | compute_bound | 21 | 3 | 3 | direct/scalar | direct/vec | 38.773 | 12.380 | 3.132 | real |
| `Winograd_Convolution_2D_padding` | Convolution | compute_bound | 23 | 3 | 3 | direct/scalar | gemm/vec | 1.466 | 0.550 | 2.665 | real |
| `DeconvolutionDepthwise_2D_kernel` | Convolution | memory_bound | 24 | 2 | 3 | nchw/none | nchw/single | 1.230 | 0.487 | 2.528 | real |
| `DeconvolutionDepthwise_2D_stride` | Convolution | memory_bound | 21 | 2 | 3 | nchw/none | nchw/single | 1.125 | 0.675 | 1.666 | real |
| `Deconvolution_1D` | Convolution | compute_bound | 23 | 5 | 4 | direct/scalar | direct/vec | 0.707 | 0.426 | 1.658 | real |
| `MatMul` | Matrix | compute_bound | 20 | 3 | 2 | direct/scalar | gemm/vec | 0.432 | 0.268 | 1.617 | real |
| `ReduceMax_with_negative_values` | Reduction | memory_bound | 24 | 4 | 4 | nchw/none | nchw/none | 0.800 | 0.538 | 1.487 | real |
| `Dense_Convolution_2D_kernel` | Convolution | compute_bound | 20 | 2 | 2 | direct/scalar | gemm/vec | 217.551 | 155.321 | 1.401 | real |
| `Asinh` | Trigonometry | memory_bound | 20 | 4 | 4 | nchw/none | packed/single | 0.391 | 0.289 | 1.353 | real |
| `Asin` | Trigonometry | memory_bound | 24 | 4 | 4 | nchw/none | nhwc/none | 4.163 | 3.086 | 1.349 | real |
| `Exp` | Unary | memory_bound | 21 | 4 | 4 | nchw/none | packed/double | 2.853 | 2.161 | 1.321 | real |
| `Cosh` | Trigonometry | memory_bound | 22 | 5 | 4 | nchw/none | packed/single | 0.240 | 0.189 | 1.274 | real |
| `Sinh` | Trigonometry | memory_bound | 23 | 5 | 4 | nchw/none | nchw/single | 5.561 | 4.478 | 1.242 | real |
| `Winograd_Convolution_2D` | Convolution | compute_bound | 21 | 6 | 4 | direct/scalar | winograd/vec | 24.001 | 19.477 | 1.232 | real |
| `Cos` | Trigonometry | memory_bound | 23 | 4 | 4 | nchw/none | nchw/none | 1.695 | 1.424 | 1.190 | real |
| `Floor` | Unary | memory_bound | 21 | 5 | 5 | nchw/none | nhwc/none | 1.257 | 1.153 | 1.091 | real |
| `Clip` | Tensor | memory_bound | 22 | 3 | 3 | nchw/none | nhwc/none | 6.747 | 6.200 | 1.088 | real |
| `Round` | Unary | memory_bound | 20 | 3 | 4 | nchw/none | packed/single | 2.908 | 2.705 | 1.075 | real |
| `Atanh` | Trigonometry | memory_bound | 20 | 2 | 2 | nchw/none | nhwc/single | 0.207 | 0.201 | 1.027 | real |
| `ConvTranspose_dilations` | Convolution | compute_bound | 13 | 2 | 2 | direct/scalar | direct/scalar | 13.916 | 13.916 | 1.000 | real |
| `Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__` | Convolution | compute_bound | 18 | 3 | 3 | direct/scalar | direct/scalar | 3.629 | 3.629 | 1.000 | real |
| `Tan` | Trigonometry | memory_bound | 15 | 2 | 2 | nchw/none | nchw/none | 0.984 | 0.984 | 1.000 | real |
| `ReduceMean` | Reduction | memory_bound | 14 | 4 | 3 | nchw/none | nchw/none | 0.872 | 0.872 | 1.000 | real |
| `ReduceMin_with_negative_values` | Reduction | memory_bound | 17 | 0 | 1 | nchw/none | nchw/none | 1.221 | 1.221 | 1.000 | real |

## Covered bins per op (niche → best latency ms; ⚑=winner, ○=baseline niche)

- **Group_Convolution_2D_kernel** (3 bins): `direct/vec`=0.0331 ⚑ · `gemm/vec`=0.0526 · `direct/scalar`=0.2641 ○
- **Dense_Convolution_1D** (3 bins): `gemm/vec`=0.2927 ⚑ · `direct/vec`=0.5299 · `direct/scalar`=2.235 ○
- **Dense_Convolution_2D** (3 bins): `gemm/vec`=1.292 ⚑ · `direct/vec`=1.438 · `direct/scalar`=9.729 ○
- **Conv** (3 bins): `direct/vec`=0.0278 ⚑ · `gemm/vec`=0.0741 · `direct/scalar`=0.1755 ○
- **Group_Convolution_2D** (3 bins): `direct/vec`=0.497 ⚑ · `gemm/vec`=0.5443 · `direct/scalar`=2.171 ○
- **Strassen_Convolution_2D** (3 bins): `gemm/vec`=0.3169 ⚑ · `direct/vec`=0.7257 · `direct/scalar`=1.159 ○
- **Conv3D** (3 bins): `direct/vec`=12.38 ⚑ · `gemm/vec`=13 · `direct/scalar`=38.77 ○
- **Winograd_Convolution_2D_padding** (3 bins): `gemm/vec`=0.5499 ⚑ · `direct/vec`=1.008 · `direct/scalar`=1.466 ○
- **DeconvolutionDepthwise_2D_kernel** (3 bins): `nchw/single`=0.4867 ⚑ · `nhwc/single`=0.4869 · `nchw/none`=1.23 ○
- **DeconvolutionDepthwise_2D_stride** (3 bins): `nchw/single`=0.6754 ⚑ · `nchw/none`=1.125 ○ · `nhwc/single`=3.179
- **Deconvolution_1D** (4 bins): `direct/vec`=0.4264 ⚑ · `direct/scalar`=0.7071 ○ · `gemm/scalar`=1.06 · `gemm/vec`=2.106
- **MatMul** (2 bins): `gemm/vec`=0.2675 ⚑ · `direct/scalar`=0.4325 ○
- **ReduceMax_with_negative_values** (4 bins): `nchw/none`=0.5383 ⚑ · `nhwc/single`=1.083 · `nchw/double`=2.136 · `nchw/single`=2.706
- **Dense_Convolution_2D_kernel** (2 bins): `gemm/vec`=155.3 ⚑ · `direct/scalar`=217.6 ○
- **Asinh** (4 bins): `packed/single`=0.2891 ⚑ · `nhwc/none`=0.3644 · `nchw/single`=0.3721 · `nchw/none`=0.3878 ○
- **Asin** (4 bins): `nhwc/none`=3.086 ⚑ · `nchw/single`=3.125 · `nchw/none`=3.398 ○ · `packed/single`=3.829
- **Exp** (4 bins): `packed/double`=2.161 ⚑ · `packed/single`=2.172 · `nhwc/none`=2.365 · `nchw/none`=2.853 ○
- **Cosh** (4 bins): `packed/single`=0.1886 ⚑ · `nhwc/single`=0.2108 · `nchw/single`=0.2401 · `nchw/none`=0.2402 ○
- **Sinh** (4 bins): `nchw/single`=4.478 ⚑ · `packed/double`=4.662 · `nhwc/none`=5.183 · `nchw/none`=5.561 ○
- **Winograd_Convolution_2D** (4 bins): `winograd/vec`=19.48 ⚑ · `direct/vec`=22.3 · `gemm/vec`=23.72 · `direct/scalar`=24 ○
- **Cos** (4 bins): `nchw/none`=1.424 ⚑ · `packed/none`=1.455 · `packed/double`=1.478 · `nhwc/single`=1.643
- **Floor** (5 bins): `nhwc/none`=1.153 ⚑ · `packed/none`=1.201 · `nhwc/single`=1.244 · `nchw/none`=1.257 ○ · `nchw/single`=1.305
- **Clip** (3 bins): `nhwc/none`=6.2 ⚑ · `nhwc/single`=6.233 · `nchw/none`=6.747 ○
- **Round** (4 bins): `packed/single`=2.705 ⚑ · `nchw/single`=2.775 · `nhwc/single`=2.804 · `nchw/none`=2.908 ○
- **Atanh** (2 bins): `nhwc/single`=0.2013 ⚑ · `nchw/none`=0.2067 ○
- **ConvTranspose_dilations** (2 bins): `direct/scalar`=13.92 ⚑ · `gemm/vec`=29.38
- **Deconvolution_2D_asymmetric_input_square_kernel___dilated____padded____strided__** (3 bins): `direct/scalar`=3.629 ⚑ · `gemm/scalar`=3.661 · `gemm/vec`=3.726
- **Tan** (2 bins): `nchw/none`=0.9838 ⚑ · `nchw/single`=0.9873
- **ReduceMean** (3 bins): `nchw/none`=0.8721 ⚑ · `nchw/double`=0.9952 · `nchw/single`=1.786
- **ReduceMin_with_negative_values** (1 bins): `nchw/none`=1.221 ⚑
