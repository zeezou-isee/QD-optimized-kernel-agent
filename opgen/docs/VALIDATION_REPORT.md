# graph_agent 环境搭建 + pnnx 编译 + LayerNorm3D 端到端验证报告

- 日期:2026-06-17
- 机器:macOS 26.5.1 / Apple Silicon (arm64)
- 工作目录:`/Users/xingze/Documents/project/kernelgen`
- 结论:**venv 安装 torch ✅ · pnnx 编译通过 ✅ · graph_agent 在 LayerNorm3D 上跑通完整计算图转换流程(round 0 一次通过,数值 allclose 通过)✅**

---

## 1. 环境搭建(本机 venv,非 conda)

不可用 conda,改用系统 Python 的 `venv`,并用 PyPI 的 `cmake` wheel(自带二进制,无需 brew/conda)。

```bash
cd EndtoEndMobilekernelAgent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch numpy openai cmake
.venv/bin/python -m pip install ncnn        # 数值验证需要 pyncnn
```

装好的版本(`.venv`):

| 包 | 版本 |
|---|---|
| torch | 2.12.0 (CPU, arm64) |
| numpy | 2.4.6 |
| ncnn (pyncnn) | 1.0.20260526 |
| openai | 2.42.0 |
| cmake | 4.3.2 (pip wheel) |

编译器:Apple clang 21.0.0 + Xcode CommandLineTools(系统自带);`make` 系统自带。

> 安装过程中遇到一次网络 `BrokenPipeError` 导致中断,加 `--retries 10 --timeout 120` 重试后成功。

---

## 2. pnnx 编译验证 ✅

pnnx 用 `PNNXProbeForPyTorchInstall()` 通过 Python 找 libtorch;为稳妥起见显式传 `-DTorch_INSTALL_DIR` 指向 venv 的 torch,并指定 venv 的 Python。

```bash
cd ncnn/tools/pnnx
TORCH=.../.venv/lib/python3.12/site-packages/torch
cmake -S . -B build -DTorch_INSTALL_DIR=$TORCH -DPython3_EXECUTABLE=.../.venv/bin/python
cmake --build build -j$(sysctl -n hw.ncpu)
```

- **Configure**:成功。`Found Torch 2.12.0`;无 TorchVision(警告,不影响);无 protobuf → 不编 onnx2pnnx(对 TorchScript→ncnn 路径无影响)。
- **Build**:`[100%] Built target pnnx`,产物 `build/src/pnnx`(7.8 MB,Mach-O arm64,ad-hoc 签名)。
- **运行验证**:`pnnx ln3d.pt inputshape=[1,8,32,32]` 成功转换,产出 `.pnnx.param/.bin`、`.ncnn.param/.bin`、`*_ncnn.py`。

### ⚠️ 发现并修复:macOS 首次运行 SIGKILL
新链接的二进制**首次**运行被 macOS 代码签名校验 `Killed: 9`(exit 137);其库依赖(libtorch 等)经 rpath 已正确加载,二次运行即正常。已在 `graph_pipeline.py` 加两处健壮性:
- `build_pnnx()`:编译后做一次 **warm-up 运行**吸收首次 kill;
- `run_conversion()`:遇 exit 137 **自动重试一次**。

---

## 3. graph_agent LayerNorm3D 端到端验证 ✅

### 运行方式
> **重要说明**:本环境 `OPENROUTER_API_KEY` 未设置,无法调用在线 LLM。为验证 agent 全流程机制,使用**桩 LLM(canned 回复)**驱动真实的 `GraphAgent.run()`(脚本 `validate_graph_agent_stub.py`)。canned 回复内容基于 ncnn 现有 `nn_LayerNorm` pass(改唯一类名 `nn_LayerNorm3d`、优先级 19 以避免符号冲突)。
>
> 因此本次验证覆盖了 agent 的**全部环节,除 LLM 文本生成质量**:analyzer → coder → 注入(新 `.cpp` + patch 两个 CMakeLists)→ 重编 pnnx → trace → 转换 → 结构验证 → 数值验证(ctest allclose)→ 回滚。

```bash
.venv/bin/python validate_graph_agent_stub.py
```

### 输入
数据集模型 `dataset/.../Normalization/LayerNorm_3d.py`:`nn.LayerNorm(normalized_shape=(32,32), eps=1e-5, affine=True)`,输入 `(1,8,32,32)`。

### 各阶段结果(round 0 一次通过)
```
[round 0] phase=identify_and_generate ok=True
          inject=True build=True convert=True structural=True numeric=True
summary: status=success  rounds=1  kept_changes=False
```

| 阶段 | 结果 | 证据 |
|---|---|---|
| identify(analyzer) | ✅ | `op_profile.json`:target_ncnn_layer=LayerNorm, needs_weight=true |
| generate(coder) | ✅ | 产出 `pass_ncnn/nn_LayerNorm3d.cpp` + `tests/ncnn/test_nn_LayerNorm3d.py` |
| inject | ✅ | 新 `.cpp` 写入 + `src/CMakeLists.txt` SRCS、`tests/ncnn/CMakeLists.txt` add_test 均被 patch |
| build | ✅ | `build.log`:`Building CXX object .../pass_ncnn/nn_LayerNorm3d.cpp.o` → `Built target pnnx` |
| convert | ✅ | 产出 `LayerNorm_3d.ncnn.param/.bin` |
| structural | ✅ | `.pnnx.param` 无 aten/prim 残留;`.ncnn.param` 含目标层 LayerNorm |
| numeric | ✅ | `ctest test_ncnn_nn_LayerNorm3d` → **Passed**(PyTorch vs ncnn allclose 1e-3) |

### 转换产物(`runs/LayerNorm_3d/round_00/LayerNorm_3d.ncnn.param`)
```
7767517
2 2
Input        in0    0 1 in0
LayerNorm    ln_0   1 1 in0 out0 0=1024 1=1.000000e-5 2=1
```
`0=affine_size=32×32=1024`、`1=eps=1e-5`、`2=affine=1`,与 PyTorch 语义一致。

### 数值验证(ctest)
```
1/1 Test #628: test_ncnn_nn_LayerNorm3d ......... Passed  0.92 sec
100% tests passed, 0 tests failed out of 1
```

### 源码树回滚
`keep_changes_on_success=False`(默认),结束后自动回滚:`pass_ncnn/nn_LayerNorm3d.cpp` 已删除,两个 CMakeLists 0 残留 —— **ncnn 源码树干净**。

### 产物目录
```
runs/LayerNorm_3d/
  analyzer.md  op_profile.json  config.json  memory.json  history.json  summary.json
  round_00/  prompt.md  response.md  result.json  build.log  numeric.log
            LayerNorm_3d.pt  *.pnnx.param/.bin  *.ncnn.param/.bin  *_ncnn.py  _trace.py
```

---

## 4. 重要说明与边界

1. **LayerNorm 在 ncnn 中本就有转换支持**(`nn_LayerNorm.cpp` + `layernorm` kernel)。因此本次属于**回归式验证**:证明(a)环境与 pnnx 可用,(b)agent 全流程机制端到端打通,(c)agent 注入的 pass 能正确编译、转换、并通过数值对齐。它**不**证明"对一个 ncnn 完全不支持的新算子,LLM 能凭空写对 pass"——那需要在线 LLM + 一个真未支持的算子。
2. **LLM 生成环节为桩**:因无 `OPENROUTER_API_KEY`。真实在线运行只需设置 key:
   ```bash
   export OPENROUTER_API_KEY=...
   cd EndtoEndMobilekernelAgent
   # 让 pnnx/ctest 用 venv 的 python/torch/ncnn:
   export PATH=$PWD/.venv/bin:$PATH
   .venv/bin/python run_graph_agent.py --task LayerNorm_3d \
       --torch-install-dir $PWD/.venv/lib/python3.12/site-packages/torch
   # 可加 --no-numeric 先只验证结构(不依赖 pyncnn)
   ```
3. **数据集 `LayerNorm_3d.py` 的小瑕疵**:docstring 说"对最后三维归一化",但 `get_init_inputs` 实际给 `(32,32)`(最后两维)。验证按文件实际语义进行,不影响结论。
4. **代码改动**:本次为适配 macOS,给 `graph_pipeline.py` 增加了 build 后 warm-up 与转换 137 重试两处健壮性改进(已纳入仓库)。

---

## 5. 复现实验命令汇总

```bash
# 1) 环境
cd EndtoEndMobilekernelAgent
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy openai cmake ncnn

# 2) 编译 pnnx
cd ../ncnn/tools/pnnx
T=$PWD/../../../EndtoEndMobilekernelAgent/.venv/lib/python3.12/site-packages/torch
cmake -S . -B build -DTorch_INSTALL_DIR=$T -DPython3_EXECUTABLE=$(dirname $T)/../../../bin/python
cmake --build build -j

# 3) 跑 graph_agent(桩 LLM 版)
cd ../../../EndtoEndMobilekernelAgent
export PATH=$PWD/.venv/bin:$PATH
.venv/bin/python validate_graph_agent_stub.py
```
