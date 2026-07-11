# ncnn convolution dispatch heuristics (auto-distilled)

- backend: **arm**  |  algo_family axis: `direct, gemm, winograd, fft, dw`
- families: **3**  |  warnings: **0**

> family-selection gates only; direct micro-variants omitted. fft is never selected by ncnn (kept as an explore-only niche).

## conv  (`Convolution`, ncnn/src/layer/arm/convolution_arm.cpp)

- **winograd** (prio 1, fp32) — kernel==3x3 && stride==1 && dilation==1 && (num_input>=8 || num_output>=8)
  - `ncnn/src/layer/arm/convolution_arm.cpp:206`  ncnn_fn=`conv3x3s1_winograd{23,43,63}`
  - tiles: `{'winograd63_channel_max': 128, 'winograd43_channel_min': 8}`
- **gemm** (prio 2, fp32) — kernel==1x1 (pointwise degenerates to GEMM)
  - `ncnn/src/layer/arm/convolution_arm.cpp:259`  ncnn_fn=`convolution_im2col_gemm`
- **gemm** (prio 3, fp32) — prefer_sgemm: work > L2_cache OR (num_input>16 || num_output>16)
  - `ncnn/src/layer/arm/convolution_arm.cpp:223`  ncnn_fn=`convolution_im2col_gemm`
- **direct** (prio 9, fp32) — fallback: small channels, stride>2, or kernels without a gemm/winograd path
  - `ncnn/src/layer/arm/convolution_arm.cpp (direct-kernel cascade)`  ncnn_fn=`conv{1x1,3x3,5x5,7x7}s{1,2}*_neon (direct cascade)`
- **winograd** (prio 1, fp16) — kernel==3x3 && stride==1 && dilation==1 && (num_input>=16 || num_output>=16)
  - `ncnn/src/layer/arm/convolution_arm_asimdhp.cpp:89`  ncnn_fn=`conv3x3s1_winograd{23,43,63}`
  - tiles: `{'winograd63_channel_max': 128, 'winograd43_channel_min': 16}`
- **gemm** (prio 3, fp16) — prefer_sgemm: work > L2_cache OR (num_input>16 || num_output>16)
  - `ncnn/src/layer/arm/convolution_arm_asimdhp.cpp:111`  ncnn_fn=`convolution_im2col_gemm`

## conv_dw  (`ConvolutionDepthWise`, ncnn/src/layer/arm/convolutiondepthwise_arm.cpp)

- **dw** (prio 1, any) — depthwise conv → specialized depthwise kernels (3x3/5x5 × s1/s2 × pack)
  - `ncnn/src/layer/arm/convolutiondepthwise_arm.cpp:130`  ncnn_fn=`convdw{3x3,5x5}s{1,2}[_pack4/_pack8]_neon`

## deconv  (`Deconvolution`, ncnn/src/layer/arm/deconvolution_arm.cpp)

- **gemm** (prio 1, any) — use_sgemm_convolution → gemm (via child Gemm layer)
  - `ncnn/src/layer/arm/deconvolution_arm.cpp:73`  ncnn_fn=`gemm`
- **direct** (prio 9, any) — fallback: direct deconv (3x3/4x4 × s1/s2, pack1)
  - `ncnn/src/layer/arm/deconvolution_arm.cpp`  ncnn_fn=`deconv{3x3,4x4}s{1,2}_neon`

