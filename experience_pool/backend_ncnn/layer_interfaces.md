# ncnn built-in layer interfaces (auto-generated)

- Layers parsed: **110**
- With doc-table cross-check **MISMATCH**: **33**
- Not present in operation-param-weight-table.md: **36**
- With parse warnings: **14**

## ⚠ MISMATCH ops (review these first)

### Convolution  (convolution.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=8` **int8_scale_term** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=19` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
  - `[2]` **weight_data_int8_scales** size=`num_output` flag=1 *(if int8_scale_term)*
  - `[3]` **bottom_blob_int8_scales** size=`1` flag=1 *(if int8_scale_term)*
  - `[4]` **top_blob_int8_scales** size=`1` flag=1 *(if int8_scale_term > 100)*
- ⚠ doc-mismatch:
  - `{'type': 'doc_only', 'id': 17, 'name': 'impl_type'}`
  - `{'type': 'src_only', 'id': 19, 'name': 'dynamic_weight'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### ConvolutionDepthWise  (convolutiondepthwise.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=8` **int8_scale_term** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=19` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
  - `[2]` **weight_data_int8_scales** size=`group` flag=1 *(if int8_scale_term == 1 || int8_scale_term == 101)*
  - `[3]` **bottom_blob_int8_scales** size=`1` flag=1 *(if int8_scale_term == 1 || int8_scale_term == 101)*
  - `[4]` **weight_data_int8_scales** size=`1` flag=1
  - `[5]` **bottom_blob_int8_scales** size=`1` flag=1
  - `[6]` **top_blob_int8_scales** size=`1` flag=1 *(if int8_scale_term > 100)*
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 19, 'name': 'dynamic_weight'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### Crop  (crop.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **woffset** = `0`
  - `id=1` **hoffset** = `0`
  - `id=2` **coffset** = `0`
  - `id=3` **outw** = `0`
  - `id=4` **outh** = `0`
  - `id=5` **outc** = `0`
  - `id=6` **woffset2** = `0`
  - `id=7` **hoffset2** = `0`
  - `id=8` **coffset2** = `0`
  - `id=9` **starts** = `Mat()`
  - `id=10` **ends** = `Mat()`
  - `id=11` **axes** = `Mat()`
  - `id=13` **doffset** = `0`
  - `id=14` **outd** = `0`
  - `id=15` **doffset2** = `0`
  - `id=19` **starts_expr** = `""`
  - `id=20` **ends_expr** = `""`
  - `id=21` **axes_expr** = `""`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 13, 'name': 'doffset'}`
  - `{'type': 'src_only', 'id': 14, 'name': 'outd'}`
  - `{'type': 'src_only', 'id': 15, 'name': 'doffset2'}`
  - `{'type': 'src_only', 'id': 19, 'name': 'starts_expr'}`
  - `{'type': 'src_only', 'id': 20, 'name': 'ends_expr'}`
  - `{'type': 'src_only', 'id': 21, 'name': 'axes_expr'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)
  - load_param may toggle one_blob_only to false depending on params (see source)

### Deconvolution  (deconvolution.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=19` **output_pad_bottom** = `output_pad_right` *(var default)*
  - `id=20` **output_w** = `0`
  - `id=21` **output_h** = `output_w` *(var default)*
  - `id=28` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 28, 'name': 'dynamic_weight'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### DeconvolutionDepthWise  (deconvolutiondepthwise.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=19` **output_pad_bottom** = `output_pad_right` *(var default)*
  - `id=20` **output_w** = `0`
  - `id=21` **output_h** = `output_w` *(var default)*
  - `id=28` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 28, 'name': 'dynamic_weight'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### Dequantize  (dequantize.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **scale_data_size** = `1`
  - `id=1` **bias_data_size** = `0`
- weights (load order):
  - `[0]` **scale_data** size=`scale_data_size` flag=1
  - `[1]` **bias_data** size=`bias_data_size` flag=1 *(if bias_data_size)*
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'scale', 'src': 'scale_data_size'}`
  - `{'type': 'name_diff', 'id': 1, 'doc': 'bias_term', 'src': 'bias_data_size'}`
  - `{'type': 'doc_only', 'id': 2, 'name': 'bias_data_size'}`

### DetectionOutput  (detectionoutput.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_class** = `0`
  - `id=1` **nms_threshold** = `0.05f`
  - `id=2` **nms_top_k** = `300`
  - `id=3` **keep_top_k** = `100`
  - `id=4` **confidence_threshold** = `0.5f`
- ⚠ doc-mismatch:
  - `{'type': 'doc_only', 'id': 5, 'name': 'variances[0]'}`
  - `{'type': 'doc_only', 'id': 6, 'name': 'variances[1]'}`
  - `{'type': 'doc_only', 'id': 7, 'name': 'variances[2]'}`
  - `{'type': 'doc_only', 'id': 8, 'name': 'variances[3]'}`

### Embed  (embed.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **input_dim** = `0`
  - `id=2` **bias_term** = `0`
  - `id=3` **weight_data_size** = `0`
  - `id=18` **int8_scale_term** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 18, 'name': 'int8_scale_term'}`

### ExpandDims  (expanddims.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=3` **axes** = `Mat()`
- ⚠ doc-mismatch:
  - `{'type': 'doc_only', 'id': 0, 'name': 'expand_w'}`
  - `{'type': 'doc_only', 'id': 1, 'name': 'expand_h'}`
  - `{'type': 'doc_only', 'id': 2, 'name': 'expand_c'}`

### Input  (input.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **w** = `0`
  - `id=1` **h** = `0`
  - `id=2` **c** = `0`
  - `id=11` **d** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 11, 'name': 'd'}`

### InstanceNorm  (instancenorm.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **channels** = `0`
  - `id=1` **eps** = `0.001f`
  - `id=2` **affine** = `1`
- weights (load order):
  - `[0]` **gamma_data** size=`channels` flag=1
  - `[1]` **beta_data** size=`channels` flag=1
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 2, 'name': 'affine'}`

### Interp  (interp.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **resize_type** = `0`
  - `id=1` **height_scale** = `1.f`
  - `id=2` **width_scale** = `1.f`
  - `id=3` **output_height** = `0`
  - `id=4` **output_width** = `0`
  - `id=5` **dynamic_target_size** = `0`
  - `id=6` **align_corner** = `0`
  - `id=9` **size_expr** = `""`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 5, 'name': 'dynamic_target_size'}`
  - `{'type': 'src_only', 'id': 6, 'name': 'align_corner'}`
  - `{'type': 'src_only', 'id': 9, 'name': 'size_expr'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)
  - load_param may toggle one_blob_only to false depending on params (see source)

### LSTM  (lstm.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **weight_data_size** = `0`
  - `id=2` **direction** = `0`
  - `id=3` **hidden_size** = `num_output` *(var default)*
  - `id=8` **int8_scale_term** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 3, 'name': 'hidden_size'}`
  - `{'type': 'src_only', 'id': 8, 'name': 'int8_scale_term'}`

### MemoryData  (memorydata.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **w** = `0`
  - `id=1` **h** = `0`
  - `id=2` **c** = `0`
  - `id=11` **d** = `0`
  - `id=21` **load_type** = `1`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 11, 'name': 'd'}`
  - `{'type': 'src_only', 'id': 21, 'name': 'load_type'}`

### MultiHeadAttention  (multiheadattention.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **embed_dim** = `0`
  - `id=1` **num_heads** = `1`
  - `id=2` **weight_data_size** = `0`
  - `id=3` **kdim** = `embed_dim` *(var default)*
  - `id=4` **vdim** = `embed_dim` *(var default)*
  - `id=5` **attn_mask** = `0`
  - `id=6` **scale** = `1.f / sqrtf(embed_dim / num_heads)`
  - `id=7` **kv_cache** = `0`
  - `id=18` **int8_scale_term** = `0`
- weights (load order):
  - `[0]` **q_weight_data** size=`embed_dim * qdim` flag=0
  - `[1]` **q_bias_data** size=`embed_dim` flag=1
  - `[2]` **k_weight_data** size=`embed_dim * kdim` flag=0
  - `[3]` **k_bias_data** size=`embed_dim` flag=1
  - `[4]` **v_weight_data** size=`embed_dim * vdim` flag=0
  - `[5]` **v_bias_data** size=`embed_dim` flag=1
  - `[6]` **out_weight_data** size=`qdim * embed_dim` flag=0
  - `[7]` **out_bias_data** size=`qdim` flag=1
  - `[8]` **q_weight_data_int8_scales** size=`embed_dim` flag=1 *(if int8_scale_term)*
  - `[9]` **k_weight_data_int8_scales** size=`embed_dim` flag=1 *(if int8_scale_term)*
  - `[10]` **v_weight_data_int8_scales** size=`embed_dim` flag=1 *(if int8_scale_term)*
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 1, 'doc': 'num_head', 'src': 'num_heads'}`
  - `{'type': 'src_only', 'id': 3, 'name': 'kdim'}`
  - `{'type': 'src_only', 'id': 4, 'name': 'vdim'}`
  - `{'type': 'src_only', 'id': 5, 'name': 'attn_mask'}`
  - `{'type': 'src_only', 'id': 6, 'name': 'scale'}`
  - `{'type': 'src_only', 'id': 7, 'name': 'kv_cache'}`
  - `{'type': 'src_only', 'id': 18, 'name': 'int8_scale_term'}`

### Packing  (packing.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **out_elempack** = `1`
  - `id=1` **use_padding** = `0`
  - `id=2` **cast_type_from** = `0`
  - `id=3` **cast_type_to** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'out_packing', 'src': 'out_elempack'}`
  - `{'type': 'doc_only', 'id': 4, 'name': 'storage_type_from'}`
  - `{'type': 'doc_only', 'id': 5, 'name': 'storage_type_to'}`

### PixelShuffle  (pixelshuffle.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **upscale_factor** = `1`
  - `id=1` **mode** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 1, 'name': 'mode'}`

### Pooling  (pooling.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **pooling_type** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **stride_w** = `1`
  - `id=3` **pad_left** = `0`
  - `id=4` **global_pooling** = `0`
  - `id=5` **pad_mode** = `0`
  - `id=6` **avgpool_count_include_pad** = `0`
  - `id=7` **adaptive_pooling** = `0`
  - `id=8` **out_w** = `0`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **stride_h** = `stride_w` *(var default)*
  - `id=13` **pad_top** = `pad_left` *(var default)*
  - `id=14` **pad_right** = `pad_left` *(var default)*
  - `id=15` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **out_h** = `out_w` *(var default)*
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'pooling_type(0: max 1: avg)', 'src': 'pooling_type'}`
  - `{'type': 'src_only', 'id': 6, 'name': 'avgpool_count_include_pad'}`
  - `{'type': 'src_only', 'id': 7, 'name': 'adaptive_pooling'}`
  - `{'type': 'src_only', 'id': 8, 'name': 'out_w'}`
  - `{'type': 'src_only', 'id': 18, 'name': 'out_h'}`

### PriorBox  (priorbox.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **min_sizes** = `Mat()`
  - `id=1` **max_sizes** = `Mat()`
  - `id=2` **aspect_ratios** = `Mat()`
  - `id=7` **flip** = `1`
  - `id=8` **clip** = `0`
  - `id=9` **image_width** = `0`
  - `id=10` **image_height** = `0`
  - `id=11` **step_width** = `-233.f`
  - `id=12` **step_height** = `-233.f`
  - `id=13` **offset** = `0.f`
  - `id=14` **step_mmdetection** = `0`
  - `id=15` **center_mmdetection** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'doc_only', 'id': 3, 'name': 'varainces[0]'}`
  - `{'type': 'doc_only', 'id': 4, 'name': 'varainces[1]'}`
  - `{'type': 'doc_only', 'id': 5, 'name': 'varainces[2]'}`
  - `{'type': 'doc_only', 'id': 6, 'name': 'varainces[3]'}`

### Proposal  (proposal.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **feat_stride** = `16`
  - `id=1` **base_size** = `16`
  - `id=2` **pre_nms_topN** = `6000`
  - `id=3` **after_nms_topN** = `300`
  - `id=4` **nms_thresh** = `0.7f`
  - `id=5` **min_size** = `16`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 4, 'doc': 'num_thresh', 'src': 'nms_thresh'}`

### Quantize  (quantize.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **scale_data_size** = `1`
- weights (load order):
  - `[0]` **scale_data** size=`scale_data_size` flag=1
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'scale', 'src': 'scale_data_size'}`

### Reduction  (reduction.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **operation** = `0`
  - `id=1` **reduce_all** = `1`
  - `id=2` **coeff** = `1.f`
  - `id=3` **axes** = `Mat()`
  - `id=4` **keepdims** = `0`
  - `id=5` **fixbug0** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 1, 'doc': 'dim', 'src': 'reduce_all'}`
  - `{'type': 'src_only', 'id': 5, 'name': 'fixbug0'}`

### Reorg  (reorg.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **stride** = `1`
  - `id=1` **mode** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 1, 'name': 'mode'}`

### Requantize  (requantize.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **scale_in_data_size** = `1`
  - `id=1` **scale_out_data_size** = `1`
  - `id=2` **bias_data_size** = `0`
  - `id=3` **activation_type** = `0`
  - `id=4` **activation_params** = `Mat()`
- weights (load order):
  - `[0]` **scale_in_data** size=`scale_in_data_size` flag=1
  - `[1]` **scale_out_data** size=`scale_out_data_size` flag=1
  - `[2]` **bias_data** size=`bias_data_size` flag=1 *(if bias_data_size)*
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'scale_in', 'src': 'scale_in_data_size'}`
  - `{'type': 'name_diff', 'id': 1, 'doc': 'scale_out', 'src': 'scale_out_data_size'}`
  - `{'type': 'name_diff', 'id': 2, 'doc': 'bias_term', 'src': 'bias_data_size'}`
  - `{'type': 'name_diff', 'id': 3, 'doc': 'bias_data_size', 'src': 'activation_type'}`
  - `{'type': 'name_diff', 'id': 4, 'doc': 'fusion_relu', 'src': 'activation_params'}`

### Reshape  (reshape.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **w** = `-233`
  - `id=1` **h** = `-233`
  - `id=2` **c** = `-233`
  - `id=6` **shape_expr** = `""`
  - `id=11` **d** = `-233`
- ⚠ doc-mismatch:
  - `{'type': 'doc_only', 'id': 3, 'name': 'permute'}`
  - `{'type': 'src_only', 'id': 6, 'name': 'shape_expr'}`
  - `{'type': 'src_only', 'id': 11, 'name': 'd'}`
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### RNN  (rnn.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **weight_data_size** = `0`
  - `id=2` **direction** = `0`
  - `id=8` **int8_scale_term** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 8, 'name': 'int8_scale_term'}`

### ShuffleChannel  (shufflechannel.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **group** = `1`
  - `id=1` **reverse** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 1, 'name': 'reverse'}`

### Slice  (slice.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **slices** = `Mat()`
  - `id=1` **axis** = `0`
  - `id=2` **indices** = `Mat()`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 2, 'name': 'indices'}`

### Softmax  (softmax.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **axis** = `0`
  - `id=1` **fixbug0** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 1, 'name': 'fixbug0'}`

### Squeeze  (squeeze.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **squeeze_w** = `0`
  - `id=1` **squeeze_h** = `0`
  - `id=2` **squeeze_c** = `0`
  - `id=3` **axes** = `Mat()`
  - `id=11` **squeeze_d** = `0`
- ⚠ doc-mismatch:
  - `{'type': 'src_only', 'id': 11, 'name': 'squeeze_d'}`

### Tile  (tile.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **axis** = `0`
  - `id=1` **tiles** = `1`
  - `id=2` **repeats** = `Mat()`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 0, 'doc': 'dim', 'src': 'axis'}`
  - `{'type': 'src_only', 'id': 2, 'name': 'repeats'}`

### YoloDetectionOutput  (yolodetectionoutput.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=True
- params:
  - `id=0` **num_class** = `20`
  - `id=1` **num_box** = `5`
  - `id=2` **confidence_threshold** = `0.01f`
  - `id=3` **nms_threshold** = `0.45f`
  - `id=4` **biases** = `Mat()`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 3, 'doc': 'num_threshold', 'src': 'nms_threshold'}`

### Yolov3DetectionOutput  (yolov3detectionoutput.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_class** = `20`
  - `id=1` **num_box** = `5`
  - `id=2` **confidence_threshold** = `0.01f`
  - `id=3` **nms_threshold** = `0.45f`
  - `id=4` **biases** = `Mat()`
  - `id=5` **mask** = `Mat()`
  - `id=6` **anchors_scale** = `Mat()`
- ⚠ doc-mismatch:
  - `{'type': 'name_diff', 'id': 3, 'doc': 'num_threshold', 'src': 'nms_threshold'}`


## All layers

### AbsVal  (absval.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### ArgMax  (argmax.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **out_max_val** = `0`
  - `id=1` **topk** = `1`

### BatchNorm  (batchnorm.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **channels** = `0`
  - `id=1` **eps** = `0.f`
- weights (load order):
  - `[0]` **slope_data** size=`channels` flag=1
  - `[1]` **mean_data** size=`channels` flag=1
  - `[2]` **var_data** size=`channels` flag=1
  - `[3]` **bias_data** size=`channels` flag=1

### Bias  (bias.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **bias_data_size** = `0`
- weights (load order):
  - `[0]` **bias_data** size=`bias_data_size` flag=1

### BinaryOp  (binaryop.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **op_type** = `0`
  - `id=1` **with_scalar** = `0`
  - `id=2` **b** = `0.f`
- parse warnings:
  - load_param may toggle one_blob_only to true depending on params (see source)
  - load_param may toggle support_inplace to true depending on params (see source)

### BNLL  (bnll.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### Cast  (cast.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **type_from** = `0`
  - `id=1` **type_to** = `0`

### CELU  (celu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **alpha** = `1.f`

### Clip  (clip.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **min** = `-FLT_MAX`
  - `id=1` **max** = `FLT_MAX`

### Concat  (concat.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **axis** = `0`

### Convolution1D  (convolution1d.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=19` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### Convolution3D  (convolution3d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=17` **pad_behind** = `pad_front` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=21` **kernel_d** = `kernel_w` *(var default)*
  - `id=22` **dilation_d** = `dilation_w` *(var default)*
  - `id=23` **stride_d** = `stride_w` *(var default)*
  - `id=24` **pad_front** = `pad_left` *(var default)*
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*

### ConvolutionDepthWise1D  (convolutiondepthwise1d.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=19` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### ConvolutionDepthWise3D  (convolutiondepthwise3d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=17` **pad_behind** = `pad_front` *(var default)*
  - `id=18` **pad_value** = `0.f`
  - `id=21` **kernel_d** = `kernel_w` *(var default)*
  - `id=22` **dilation_d** = `dilation_w` *(var default)*
  - `id=23` **stride_d** = `stride_w` *(var default)*
  - `id=24` **pad_front** = `pad_left` *(var default)*
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*

### CopyTo  (copyto.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **woffset** = `0`
  - `id=1` **hoffset** = `0`
  - `id=2` **coffset** = `0`
  - `id=9` **starts** = `Mat()`
  - `id=11` **axes** = `Mat()`
  - `id=13` **doffset** = `0`

### CumulativeSum  (cumulativesum.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **axis** = `0`

### Deconvolution1D  (deconvolution1d.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=20` **output_w** = `0`
  - `id=28` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### Deconvolution3D  (deconvolution3d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=17` **pad_behind** = `pad_front` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=19` **output_pad_bottom** = `output_pad_right` *(var default)*
  - `id=20` **output_pad_behind** = `output_pad_right` *(var default)*
  - `id=21` **kernel_d** = `kernel_w` *(var default)*
  - `id=22` **dilation_d** = `dilation_w` *(var default)*
  - `id=23` **stride_d** = `stride_w` *(var default)*
  - `id=24` **pad_front** = `pad_left` *(var default)*
  - `id=25` **output_w** = `0`
  - `id=26` **output_h** = `output_w` *(var default)*
  - `id=27` **output_d** = `output_w` *(var default)*
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*

### DeconvolutionDepthWise1D  (deconvolutiondepthwise1d.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=20` **output_w** = `0`
  - `id=28` **dynamic_weight** = `0`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### DeconvolutionDepthWise3D  (deconvolutiondepthwise3d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=7` **group** = `1`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=17` **pad_behind** = `pad_front` *(var default)*
  - `id=18` **output_pad_right** = `0`
  - `id=19` **output_pad_bottom** = `output_pad_right` *(var default)*
  - `id=20` **output_pad_behind** = `output_pad_right` *(var default)*
  - `id=21` **kernel_d** = `kernel_w` *(var default)*
  - `id=22` **dilation_d** = `dilation_w` *(var default)*
  - `id=23` **stride_d** = `stride_w` *(var default)*
  - `id=24` **pad_front** = `pad_left` *(var default)*
  - `id=25` **output_w** = `0`
  - `id=26` **output_h** = `output_w` *(var default)*
  - `id=27` **output_d** = `output_w` *(var default)*
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*

### DeepCopy  (deepcopy.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False

### DeformableConv2D  (deformableconv2d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=5` **bias_term** = `0`
  - `id=6` **weight_data_size** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*

### Diag  (diag.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **diagonal** = `0`

### Dropout  (dropout.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **scale** = `1.f`

### Einsum  (einsum.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **equation_mat** = `Mat()`

### Eltwise  (eltwise.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **op_type** = `0`
  - `id=1` **coeffs** = `Mat()`

### ELU  (elu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **alpha** = `0.1f`

### Erf  (erf.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### Exp  (exp.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **base** = `-1.f`
  - `id=1` **scale** = `1.f`
  - `id=2` **shift** = `0.f`

### Flatten  (flatten.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False

### Flip  (flip.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **axes** = `Mat()`

### Fold  (fold.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=20` **output_w** = `0`
  - `id=21` **output_h** = `output_w` *(var default)*

### GELU  (gelu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **fast_gelu** = `0`

### Gemm  (gemm.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **alpha** = `1.f`
  - `id=1` **beta** = `1.f`
  - `id=2` **transA** = `0`
  - `id=3` **transB** = `0`
  - `id=4` **constantA** = `0`
  - `id=5` **constantB** = `0`
  - `id=6` **constantC** = `0`
  - `id=7` **constantM** = `0`
  - `id=8` **constantN** = `0`
  - `id=9` **constantK** = `0`
  - `id=10` **constant_broadcast_type_C** = `0`
  - `id=11` **output_N1M** = `0`
  - `id=12` **output_elempack** = `0`
  - `id=13` **output_elemtype** = `0`
  - `id=14` **output_transpose** = `0`
  - `id=18` **int8_scale_term** = `0`
  - `id=20` **constant_TILE_M** = `0`
  - `id=21` **constant_TILE_N** = `0`
  - `id=22` **constant_TILE_K** = `0`
- weights (load order):
  - `[0]` **C_data** size=`1` flag=0 *(if constantC == 1 && constant_broadcast_type_C != -1)*
  - `[1]` **C_data** size=`constantM` flag=0 *(if constantC == 1 && constant_broadcast_type_C != -1)*
  - `[2]` **A_data_int8_scales** size=`constantM` flag=1 *(if constantA == 1)*
- parse warnings:
  - load_param may toggle one_blob_only to true depending on params (see source)
  - load_param may toggle one_blob_only to true depending on params (see source)
  - load_param may toggle one_blob_only to true depending on params (see source)

### GLU  (glu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **axis** = `0`

### GridSample  (gridsample.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **sample_type** = `1`
  - `id=1` **padding_mode** = `1`
  - `id=2` **align_corner** = `0`
  - `id=3` **permute_fusion** = `0`

### GroupNorm  (groupnorm.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **group** = `1`
  - `id=1` **channels** = `0`
  - `id=2` **eps** = `0.001f`
  - `id=3` **affine** = `1`
- weights (load order):
  - `[0]` **gamma_data** size=`channels` flag=1
  - `[1]` **beta_data** size=`channels` flag=1

### GRU  (gru.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **weight_data_size** = `0`
  - `id=2` **direction** = `0`
  - `id=8` **int8_scale_term** = `0`

### HardSigmoid  (hardsigmoid.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **alpha** = `0.2f`
  - `id=1` **beta** = `0.5f`

### HardSwish  (hardswish.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **alpha** = `0.2f`
  - `id=1` **beta** = `0.5f`

### InnerProduct  (innerproduct.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **num_output** = `0`
  - `id=1` **bias_term** = `0`
  - `id=2` **weight_data_size** = `0`
  - `id=8` **int8_scale_term** = `0`
  - `id=9` **activation_type** = `0`
  - `id=10` **activation_params** = `Mat()`
- weights (load order):
  - `[0]` **weight_data** size=`weight_data_size` flag=0
  - `[1]` **bias_data** size=`num_output` flag=1 *(if bias_term)*
  - `[2]` **weight_data_int8_scales** size=`num_output` flag=1 *(if int8_scale_term)*
  - `[3]` **bottom_blob_int8_scales** size=`1` flag=1 *(if int8_scale_term)*

### InverseSpectrogram  (inversespectrogram.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **n_fft** = `0`
  - `id=1` **returns** = `0`
  - `id=2` **hoplen** = `n_fft / 4`
  - `id=3` **winlen** = `n_fft` *(var default)*
  - `id=4` **window_type** = `0`
  - `id=5` **center** = `1`
  - `id=7` **normalized** = `0`

### LayerNorm  (layernorm.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **affine_size** = `0`
  - `id=1` **eps** = `0.001f`
  - `id=2` **affine** = `1`
- weights (load order):
  - `[0]` **gamma_data** size=`affine_size` flag=1
  - `[1]` **beta_data** size=`affine_size` flag=1

### Log  (log.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **base** = `-1.f`
  - `id=1` **scale** = `1.f`
  - `id=2` **shift** = `0.f`

### LRN  (lrn.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **region_type** = `0`
  - `id=1` **local_size** = `5`
  - `id=2` **alpha** = `1.f`
  - `id=3` **beta** = `0.75f`
  - `id=4` **bias** = `1.f`

### MatMul  (matmul.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **transB** = `0`

### Mish  (mish.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### MVN  (mvn.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **normalize_variance** = `0`
  - `id=1` **across_channels** = `0`
  - `id=2` **eps** = `0.0001f`

### Noop  (noop.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=True

### Normalize  (normalize.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **across_spatial** = `0`
  - `id=1` **channel_shared** = `0`
  - `id=2` **eps** = `0.0001f`
  - `id=3` **scale_data_size** = `0`
  - `id=4` **across_channel** = `1`
  - `id=9` **eps_mode** = `0`
- weights (load order):
  - `[0]` **scale_data** size=`scale_data_size` flag=1

### Padding  (padding.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **top** = `0`
  - `id=1` **bottom** = `0`
  - `id=2` **left** = `0`
  - `id=3` **right** = `0`
  - `id=4` **type** = `0`
  - `id=5` **value** = `0.f`
  - `id=6` **per_channel_pad_data_size** = `0`
  - `id=7` **front** = `0`
  - `id=8` **behind** = `0`
- weights (load order):
  - `[0]` **per_channel_pad_data** size=`per_channel_pad_data_size` flag=1 *(if per_channel_pad_data_size)*

### Permute  (permute.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **order_type** = `0`

### Pooling1D  (pooling1d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **pooling_type** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **stride_w** = `1`
  - `id=3` **pad_left** = `0`
  - `id=4` **global_pooling** = `0`
  - `id=5` **pad_mode** = `0`
  - `id=6` **avgpool_count_include_pad** = `0`
  - `id=7` **adaptive_pooling** = `0`
  - `id=8` **out_w** = `0`
  - `id=14` **pad_right** = `pad_left` *(var default)*

### Pooling3D  (pooling3d.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **pooling_type** = `0`
  - `id=1` **kernel_w** = `0`
  - `id=2` **stride_w** = `1`
  - `id=3` **pad_left** = `0`
  - `id=4` **global_pooling** = `0`
  - `id=5` **pad_mode** = `0`
  - `id=6` **avgpool_count_include_pad** = `0`
  - `id=7` **adaptive_pooling** = `0`
  - `id=8` **out_w** = `0`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **stride_h** = `stride_w` *(var default)*
  - `id=13` **pad_top** = `pad_left` *(var default)*
  - `id=14` **pad_right** = `pad_left` *(var default)*
  - `id=15` **pad_bottom** = `pad_top` *(var default)*
  - `id=16` **pad_behind** = `pad_front` *(var default)*
  - `id=18` **out_h** = `out_w` *(var default)*
  - `id=21` **kernel_d** = `kernel_w` *(var default)*
  - `id=22` **stride_d** = `stride_w` *(var default)*
  - `id=23` **pad_front** = `pad_left` *(var default)*
  - `id=28` **out_d** = `out_w` *(var default)*

### Power  (power.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **power** = `1.f`
  - `id=1` **scale** = `1.f`
  - `id=2` **shift** = `0.f`

### PReLU  (prelu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **num_slope** = `0`
- weights (load order):
  - `[0]` **slope_data** size=`num_slope` flag=1

### PSROIPooling  (psroipooling.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **pooled_width** = `7`
  - `id=1` **pooled_height** = `7`
  - `id=2` **spatial_scale** = `0.0625f`
  - `id=3` **output_dim** = `0`

### ReLU  (relu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **slope** = `0.f`

### RMSNorm  (rmsnorm.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **affine_size** = `0`
  - `id=1` **eps** = `0.001f`
  - `id=2` **affine** = `1`
- weights (load order):
  - `[0]` **gamma_data** size=`affine_size` flag=1

### ROIAlign  (roialign.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **pooled_width** = `0`
  - `id=1` **pooled_height** = `0`
  - `id=2` **spatial_scale** = `1.f`
  - `id=3` **sampling_ratio** = `0`
  - `id=4` **aligned** = `false` *(var default)*
  - `id=5` **version** = `0`

### ROIPooling  (roipooling.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **pooled_width** = `0`
  - `id=1` **pooled_height** = `0`
  - `id=2` **spatial_scale** = `1.f`

### RotaryEmbed  (rotaryembed.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=0` **interleaved** = `0`

### Scale  (scale.h)
- base class: `Layer`
- forward: 2 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **scale_data_size** = `0`
  - `id=1` **bias_term** = `0`
- weights (load order):
  - `[0]` **scale_data** size=`scale_data_size` flag=1
  - `[1]` **bias_data** size=`scale_data_size` flag=1 *(if bias_term)*
- parse warnings:
  - load_param may toggle one_blob_only to false depending on params (see source)

### SDPA  (sdpa.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False
- params:
  - `id=5` **attn_mask** = `0`
  - `id=6` **scale** = `0.f`
  - `id=7` **kv_cache** = `0`
  - `id=18` **int8_scale_term** = `0`

### SELU  (selu.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **alpha** = `1.67326324f`
  - `id=1` **lambda** = `1.050700987f`

### Shrink  (shrink.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **bias** = `0.0f`
  - `id=1` **lambd** = `0.5f`

### Sigmoid  (sigmoid.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### Softplus  (softplus.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### Spectrogram  (spectrogram.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **n_fft** = `0`
  - `id=1` **power** = `0`
  - `id=2` **hoplen** = `n_fft / 4`
  - `id=3` **winlen** = `n_fft` *(var default)*
  - `id=4` **window_type** = `0`
  - `id=5` **center** = `1`
  - `id=6` **pad_type** = `2`
  - `id=7` **normalized** = `0`
  - `id=8` **onesided** = `1`

### Split  (split.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=False support_inplace=False

### SPP  (spp.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **pooling_type** = `0`
  - `id=1` **pyramid_height** = `1`

### StatisticsPooling  (statisticspooling.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=0` **include_stddev** = `0`

### Swish  (swish.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### TanH  (tanh.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True

### Threshold  (threshold.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **threshold** = `0.f`

### UnaryOp  (unaryop.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=True
- params:
  - `id=0` **op_type** = `0`

### Unfold  (unfold.h)
- base class: `Layer`
- forward: 1 overload(s)
- flags (default): one_blob_only=True support_inplace=False
- params:
  - `id=1` **kernel_w** = `0`
  - `id=2` **dilation_w** = `1`
  - `id=3` **stride_w** = `1`
  - `id=4` **pad_left** = `0`
  - `id=11` **kernel_h** = `kernel_w` *(var default)*
  - `id=12` **dilation_h** = `dilation_w` *(var default)*
  - `id=13` **stride_h** = `stride_w` *(var default)*
  - `id=14` **pad_top** = `pad_left` *(var default)*
  - `id=15` **pad_right** = `pad_left` *(var default)*
  - `id=16` **pad_bottom** = `pad_top` *(var default)*
  - `id=18` **pad_value** = `0.f`

