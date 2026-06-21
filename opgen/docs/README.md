# graph_agent — ncnn 计算图转换生成 Agent

为 ncnn 框架**新增算子的计算图转换**(PyTorch → ncnn)自动生成、注入、构建、验证、迭代修复的独立 Agent。

> 背景与原理见 [`ncnn_graph_conversion.md`](ncnn_graph_conversion.md);设计方案见 [`graph_agent_design.md`](graph_agent_design.md)。

## 它解决什么

旧 MoKA 只能改 ncnn **已有**算子的 kernel,无法引入框架里**不存在**的新算子——因为新算子必须先有 PyTorch→ncnn 的计算图转换(PNNX pass)才能生成 `.ncnn.param/.bin`。`graph_agent` 补的就是这条链路。

## 架构

**agent loop(状态机) + 功能函数 + 三角色**,与 MoKA 同构但独立自成一套。

```
graph_agent.py     GraphAgent 类:loop(identify → generate → inject → build → convert → verify → repair)
graph_pipeline.py  功能函数(可单独调用):extract_code_blocks / retrieve_examples / inject_files /
                   build_pnnx / make_pt / run_conversion / verify_structural / verify_numeric / restore_files
graph_schemas.py   OpProfile / GraphResult / BackupHandle / GraphRound
graph_prompts.py   analyzer / coder / debugger(三模式) 角色 prompt
llm_api.py         OpenRouter LLM 封装(独立)
config.py          路径与运行参数(GraphConfig)
run_graph_agent.py CLI 入口
tools/             复用的文件/shell 工具
runs/<task>/       每个任务的产物
```

### loop 状态机(pipeline 结果驱动)
```
round 0: analyzer 识别算子(OpProfile) → 检索相似 pass → coder 写 pass+test → 跑转换流程
round 1+: 按第一个失败阶段选 debugger 模式 → coder 据反馈重写 → 重跑
   first_failure: inject → build → convert/structural → numeric
```

### 验证两级
- **结构验证**(永远做,不依赖 kernel):解析 `.pnnx.param`(无残留 `aten::/prim::`)+ `.ncnn.param`(目标 layer 出现)。
- **数值验证**(需 ncnn kernel 可运行,由别的 agent 提供):`ctest` 跑端到端 `allclose`。`--no-numeric` 可关闭。

### 出错反馈(精准定位坏在哪段)
| 阶段 | 反馈 | 指向 |
|---|---|---|
| build | 编译错误+上下文(过滤到新 pass 文件) | C++ 语法/API |
| convert | `.pnnx.param` 残留 `aten::/prim::` | pass_level2 没覆盖 |
| convert | `.ncnn.param` 缺目标 layer | pass_ncnn type_str/match 错 |
| numeric | allclose 差异/shape 不符 | write() param 编号/权重错 |

## 前置条件

1. **pnnx 从源码构建**:环境需有 PyTorch(pnnx 用 `PNNXProbeForPyTorchInstall()` 自动找 pip 的 libtorch)。首轮构建较慢,之后增量。
2. 环境变量:`export OPENROUTER_API_KEY=...`(可选 `OPENROUTER_MAX_TOKENS`)。
3. 数值验证需目标算子的 ncnn 运行时 kernel 已存在于 `ncnn/src/layer`。

## 用法

```bash
cd EndtoEndMobilekernelAgent

# 按 task 名自动定位数据集模型
python run_graph_agent.py --task HardSigmoid

# 指定 PyTorch 参考文件
python run_graph_agent.py --task MyOp --model /path/to/MyOp.py --max-rounds 8

# 只做结构验证(不依赖 kernel,闭环更快)
python run_graph_agent.py --task MyOp --no-numeric

# 成功后保留注入的 pass 文件与 CMake 改动(默认结束后回滚)
python run_graph_agent.py --task MyOp --keep-on-success
```

### 作为库单独调用
```python
from graph_agent import GraphAgent
from config import GraphConfig

agent = GraphAgent(task_name="HardSigmoid", cfg=GraphConfig(run_numeric=False))
summary = agent.run()        # -> dict(status, history, final_result, ...)
```
每个功能函数也可单独 import 验证,例如只测结构验证或只跑构建:
```python
from graph_pipeline import build_pnnx, verify_structural
```

## 产物
```
runs/<task>/
  config.json  analyzer.md  op_profile.json  memory.json  history.json  summary.json
  round_XX/  prompt.md  response.md  result.json  build.log  numeric.log  *.pnnx.param  *.ncnn.param ...
```

## 安全:源码树注入与回滚
注入会**新建** pass 文件并 **patch 两个 CMakeLists**(`src/CMakeLists.txt` 的 SRCS 列表 + `tests/ncnn/CMakeLists.txt` 的 add_test)。`BackupHandle` 记录原始状态,默认在结束/失败时 `restore_files` 完整回滚(删新建文件 + 还原 CMake),保证 ncnn 源码树干净;`--keep-on-success` 可保留成功结果。

## 当前范围(首版)
- 只针对**无权重算子(unary/functional)**;weighted(conv/linear/bn)、tensor_manip、composite 为二期。
- 结构验证 + 数值 allclose 均做。
