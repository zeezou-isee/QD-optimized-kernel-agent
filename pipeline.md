# 移动端算子测试 Pipeline 需求文档

## 1. 背景与目标

本项目的总体目标：用 agent 针对不同硬件 / 后端，为 MNN、ncnn 等移动端推理框架生成 C++ 算子实现，并尽可能提升性能。

本文档**只描述算子测试 pipeline 的搭建需求**，不涉及 agent 的工作流编排。pipeline 的职责是：给定一个算子的 Python 定义、目标框架、目标硬件与后端，能够**自动化地完成转换、正确性校验与性能测试**，为后续 agent 生成 / 优化算子提供一个可复用、可量化的评测闭环。

可以把整个 pipeline 理解为 KernelBench 在移动端推理框架上的扩展版本：
- KernelBench 评测的是 CUDA kernel；
- 这里评测的是「Python 算子定义 → 移动端框架 C++ 算子」的端到端正确性与性能。

## 2. 输入与输出

### 2.1 任务输入
- **算子的 Python 定义**：来自 `datasets/MobileKernelBench/`，KernelBench 风格（见第 3 节）。
- **目标框架**：`MNN` | `ncnn`（设计上可扩展到其他框架）。
- **目标硬件**：如 `arm64` / `x86` / 具体 SoC；当前开发机为 x86 Linux/WSL2，移动端硬件后续通过交叉编译 + 设备执行接入。
- **目标后端**：框架内的后端类型，如 MNN 的 `CPU` / `OpenCL` / `Vulkan`，ncnn 的 `CPU` / `Vulkan`。

### 2.2 任务输出
- 对应框架、硬件、后端下的**算子 C++ 代码**（path A 为替换实现，path B 为新增实现 + 计算图转换）。
- 一份**评测报告**：是否转换成功、正确性是否达标、性能数据（与原生实现对比）。

## 3. 数据集格式（已确认）

每个算子是一个独立的 `.py` 文件，遵循 KernelBench 约定：

```python
class Model(nn.Module):
    def __init__(self, ...):   # 由 get_init_inputs() 提供初始化参数
        ...
    def forward(self, x, ...): # 真正的算子语义
        ...

def get_inputs():       # 返回 forward 的输入张量列表（含权重等）
    return [...]

def get_init_inputs():  # 返回 Model.__init__ 的参数列表
    return [...]
```

要点（实现 pipeline 时必须处理）：
- `forward` 可能有**多输入**（如 Conv 的 `x, w, b`），权重作为运行时输入而非 `nn.Parameter`，导出 ONNX 时需要决定哪些是图输入、哪些固化为常量（见第 5 节转换策略）。
- 同一类算子有多种参数变体（如 `LayerNorm` / `LayerNorm_eps` / `LayerNorm_no_affine`）。
- 目录即分类：`Activation` / `Binary` / `Convolution` / `Logic` / `Matrix` / `Normalization` / `Others` / `Pooling` / `Reduction` / `Tensor` / `Trigonometry` / `Unary`，共约 190 个算子。
- 存在框架原生**不支持**的算子（如 ncnn 缺 `Stft`），这类天然走 path B。

## 4. 环境现状（已确认）

- **Python 环境**：conda 环境 `mnn` 中已安装 `torch 2.4.0+cpu`、`torchvision`、`torchaudio`、`onnx 1.20.1`、`onnxruntime 1.23.2`、`pnnx`、`numpy`。pipeline 默认使用 `mnn` 环境。
- **MNN**：已编译，`frameworks/MNN/build/` 下有 `MNNConvert`（支持 `TF/CAFFE/ONNX/TFLITE/MNN/JSON`）、`ModuleBasic.out`、`testModel.out`、`backendTest.out` 等。CPU 算子源码在 `frameworks/MNN/source/backend/cpu/CPU*.cpp`。
- **ncnn**：已编译，`frameworks/ncnn/build/` 下有 `tools/onnx`（`onnx2ncnn` 源码）、`tools/ncnnoptimize`、`benchmark/benchncnn`，并已构建 pnnx。算子源码在 `frameworks/ncnn/src/layer/*.cpp`（含 `arm/`、`vulkan/` 后端子目录）。
- **Python 绑定**：`mnn` 环境**尚未安装** pymnn / ncnn python 包。pipeline 需自行决定是用 Python 绑定还是直接调用命令行工具（见第 6 节，倾向命令行工具以减少环境耦合）。

## 5. Pipeline 总体结构

整个 pipeline 分为两大阶段：

### 阶段一：建立 baseline（框架原生实现）
对数据集中**每一个** Python 模型：
1. 加载 Python `Model` + `get_inputs` + `get_init_inputs`，跑出 PyTorch 参考输出。
2. 用框架原生路径，将 Python 模型转换为框架特定格式（`.mnn` / `.param+.bin`）。
3. 用框架原生实现做推理，**校验输出与 PyTorch 参考输出的误差是否达标**。
4. 测试该原生实现的**性能**（latency / 吞吐）。
5. 记录：转换是否成功、正确性、性能，作为该算子的 baseline。

这一阶段同时充当**框架算子支持度的探测器**：转换 / 校验成功即「框架已原生支持」，失败（缺算子、转换报错）即「未支持」。

### 阶段二：算子支持度判定与分流
根据阶段一结果及框架算子清单，判断目标框架是否支持该 Python 文件对应的算子：

- **已有原生实现 → 路径 A（优化现有算子）**
- **未实现 → 路径 B（实现新算子）**

#### 路径 A：优化现有算子
1. 用 LLM 生成的算子 C++ 代码**替换**框架中原本的算子实现（如 MNN 的 `CPURelu.cpp`、ncnn 的 `relu.cpp`）。
2. 重新编译框架（增量编译）。
3. 正确性校验（复用阶段一的参考输出与误差标准）。
4. 性能测试，与 baseline 对比。

#### 路径 B：实现新算子
1. **注册新算子**（MNN：新增 OpType + Execution + 在 CPUBackend 注册；ncnn：新增 layer + 在 layer_registry 注册）。
2. **使用 agent 实现计算图转换**——把 ONNX / PyTorch 图里的该算子映射到框架的新算子节点上。**注意：这一步的计算图转换逻辑由 agent 生成，不是 pipeline 中写死的固定步骤**。pipeline 只提供「插入 agent 生成的转换代码 → 编译 → 跑通」的机制。
3. 重新编译框架。
4. 正确性校验。
5. 性能测试。

> 路径 A / B 的「LLM/agent 生成代码」本身不在本 pipeline 范围内；pipeline 负责提供**替换 / 注册 → 编译 → 校验 → 测速**的可重复执行框架，以及把 agent 产物接入的明确接口。

## 6. 各步骤详细需求

### 6.1 Python 模型加载与参考输出
- 动态 import 任意算子 `.py`，实例化 `Model(*get_init_inputs())`，用 `get_inputs()` 跑前向，得到参考输出与所有输入张量（保存为 `.npy` / `.bin`，供框架侧推理复用同一份输入）。
- 固定随机种子，保证输入可复现。
- 记录输入 / 输出的 shape、dtype。

### 6.2 模型转换（Python → 框架格式）
统一中间表示用 **ONNX**：
- `torch.onnx.export`（或 pnnx 直接从 torch）导出 ONNX。需处理：
  - 多输入算子中权重的处理（作为图输入 vs 常量）。
  - 动态 shape 是否需要固定。
- **MNN**：`MNNConvert -f ONNX --modelFile x.onnx --MNNModel x.mnn`。
- **ncnn**：`onnx2ncnn x.onnx x.param x.bin`（可选 `ncnnoptimize`）；或走 pnnx 路径。
- 转换失败要能区分「缺算子 / 转换器 bug / 导出问题」并记录原因（用于阶段二分流）。

### 6.3 框架侧推理与正确性校验
- 用与 PyTorch 相同的输入张量喂给框架模型。
- 推理执行方式（二选一，倾向后者以降低环境耦合）：
  - Python 绑定（需先安装 pymnn / ncnn-python）；
  - 命令行工具：MNN 用 `ModuleBasic.out` / `testModel.out`（支持指定输入目录、forwardType、precision），ncnn 用自带 test 程序或最小 C++ harness。
- **误差标准**（需明确并可配置）：建议默认 `rtol=1e-3, atol=1e-4`（fp32）；fp16 / 量化场景放宽，按后端 precision 单独设阈值。校验同时输出 max abs error、cosine similarity 等指标。

### 6.4 性能测试
- 指定 forwardType（CPU/OpenCL/Vulkan）、线程数、precision、循环次数、warmup。
- 记录平均 latency、min/max、（可选）吞吐。
- baseline 与优化后用**同一测速配置**，结果可直接对比，输出加速比。

### 6.5 编译接入（路径 A / B 共用）
- 提供「替换 / 新增源文件 → 增量编译 → 产出可执行测试件」的脚本化流程。
- 编译失败需回传完整 stderr（供 agent 迭代修复）。
- 应支持隔离构建（避免污染主 build 目录），便于并行评测与回滚。

### 6.6 算子注册接口（路径 B）
- 明确 MNN / ncnn 各自注册新算子需要改动的文件清单与最小改动点，作为 agent 的「填空模板」。
- 计算图转换部分由 agent 生成，pipeline 提供：转换代码插入位置、ONNX 算子 → 框架算子的接口约定、编译与跑通验证。

## 7. 目录与产物规划（建议）

```
KernelAgent/
├── pipeline/                 # 本 pipeline 的实现代码
│   ├── runner.py             # 单算子端到端驱动
│   ├── loader.py             # 6.1 Python 模型加载 + 参考输出
│   ├── convert/              # 6.2 各框架转换封装（mnn.py / ncnn.py）
│   ├── infer/                # 6.3 各框架推理 + 校验封装
│   ├── bench/                # 6.4 性能测试封装
│   ├── build/                # 6.5 编译接入
│   ├── registry/             # 6.6 路径 B 注册模板
│   └── config.py             # 误差阈值、后端、设备等配置
├── results/                  # 评测产物
│   └── <framework>/<backend>/<op>/   # onnx、模型、输入输出、报告 json、日志
└── pipeline.md               # 本文档
```

每个算子产出一份 `report.json`：转换状态、正确性指标、baseline 与优化后性能、走的路径（A/B）、失败原因等，便于汇总成排行榜。

## 8. 已确认决策

1. **执行硬件（已定）**：**正确性校验在当前机器（x86 / WSL2）CPU 上完成；性能测试在移动端 arm64 上进行，通过 `adb` 把模型 / 测速程序推到设备执行并回收结果。** 这意味着：
   - 转换 + 正确性校验：x86 本机，用 CPU 后端跑框架推理与 PyTorch 参考对比。
   - 性能测试：需要 **arm64 交叉编译**框架的测速程序（MNN 的 `ModuleBasic.out`/`benchmark`，ncnn 的 `benchncnn`），`adb push` 模型与可执行件到设备，`adb shell` 执行并解析 latency。
   - 因此 pipeline 在编译接入（6.5）上要支持**双目标**：x86（校验用）+ arm64（测速用）。
2. **推理调用方式（已定）**：**统一走命令行工具**，不安装 pymnn / ncnn-python 绑定。MNN 用 `ModuleBasic.out` / `testModel.out`，ncnn 用自带 test 程序或最小 C++ harness；输入 / 输出通过文件交换。降低环境耦合，x86 与 arm64 两端口径一致。
3. **范围（已定）**：**第一版先覆盖少量代表性算子打通全流程**，含一个 path A 示例与一个 path B 示例，再批量铺开。

### 仍需在实现时确认（非阻塞）
- **误差阈值**：默认采用第 6.3 节建议（fp32 `rtol=1e-3, atol=1e-4`），fp16 / 量化按后端 precision 放宽。
- **转换中间表示**：默认统一走 ONNX；ncnn 侧若某算子 onnx2ncnn 支持不佳，可回退到 pnnx（PyTorch 直转）。
- **adb 设备**：需提供一台可用的 arm64 Android 设备 / 模拟器及 NDK 工具链路径。

## 9. 第一版里程碑（建议）

- M1：阶段一 baseline 闭环——任选 1 个算子（如 Relu），跑通 Python→ONNX→MNN/ncnn→推理校验→测速。
- M2：路径 A 闭环——替换该算子 C++ 实现 → 增量编译 → 校验 → 测速对比。
- M3：路径 B 闭环——选一个框架不支持的算子（如 ncnn 的 Stft），打通注册 + agent 计算图转换接入 + 编译 + 校验。
- M4：批量化——把单算子驱动扩展到整个数据集，产出汇总报告。

