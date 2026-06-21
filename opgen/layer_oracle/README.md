# layer_oracle — ncnn 单算子 oracle 验证工具(方案A)

把"得到 ncnn layer 输出"的 harness 写死成一个**泛型 runner**,被测算子只需提供 kernel 的 `.cpp/.h`。
通用 runner 与被测 `.cpp` 一起编译、链接预编好的 `libncnn.a`,直接 `new <Class>()` 调用其 forward,
**不改 ncnn 源码树、不重编 libncnn、不写 per-op 测试**。再用 PyTorch 作 oracle 做 `allclose`。

## 文件
- `layer_oracle_runner.cpp` — 泛型 runner(写死一次,`-DCANDIDATE_HEADER/-DCANDIDATE_CLASS` 指定被测算子)
- `oracle.py` — `LayerOracle` 类:编译(带缓存)/运行/`verify`(vs PyTorch)
- `__init__.py` — 导出

## 一次性前提:编一个 libncnn.a
```bash
cmake -S ncnn -B ncnn/build_lib -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF \
  -DNCNN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ncnn/build_lib -j
```

## 用法
```python
from layer_oracle import LayerOracle, torch_to_ncnn_input
import torch, torch.nn.functional as F

oc = LayerOracle()                      # 自动定位 ncnn/ 与 build_lib/
x = torch.randn(1, 2, 10); w = torch.randn(4,2,3); b = torch.randn(4)
ref = F.conv1d(x, w, b).numpy()         # PyTorch = oracle

verdict = oc.verify(
    candidate_cpp="ncnn/src/layer/convolution1d.cpp",
    class_name="Convolution1D", header="convolution1d.h",
    params={0:4,1:3,2:1,3:1,4:0,5:1,6:24},   # ncnn param-id -> 值(算子自己定义)
    inputs=[torch_to_ncnn_input(x.numpy())],  # ncnn 布局(去掉 batch 维)
    weights=[w.numpy().reshape(-1), b.numpy()],
    reference=ref, tol=1e-3,
)
print(verdict.passed, verdict.detail)   # True  max_diff=0.000000 ...
```
只跑不验证(拿输出):`oc.run(...)` → `OracleResult.outputs[0]`(numpy)。

## 约定
- **bin 协议**:`[int32 ndim][int32 dims...][float32 data]`(`write_bin/read_bin`)。
- **输入布局**:numpy 按 ncnn 单样本布局(去 batch);`torch_to_ncnn_input` 自动去掉 axis0。
  形状映射:`(N,C,H,W)→(C,H,W)`、`(N,C,L)→(C,L)`、`(N,C)→(C,)`。
- **权重**:按 `load_model` 里 `mb.load(...)` 的顺序传(如 conv:`[weight_flat, bias]`),裸 float、自动 flatten。
- **params**:`{param_id: 值}`,int/float 自动区分(float 用小数点)。
- **forward 分派**:runner 按 `one_blob_only/support_inplace` 自动选 forward / forward_inplace / 多输入。
- **编译缓存**:按 candidate 文件 mtime,改了才重编。

## 自检
```bash
python run_layer_oracle.py     # Convolution1D vs F.conv1d -> PASS, max_diff=0.000000
```

## 给 layer agent 复用
未来"从零写 kernel"的 agent:让 LLM 只产出 `mylayer.{h,cpp}` + param-id 映射,然后
`LayerOracle.verify(candidate_cpp=agent写的.cpp, ..., reference=PyTorch输出)` 即可自动验数学正确性,
全程不依赖计算图转换、不依赖 baseline。
