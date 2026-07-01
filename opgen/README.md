# opgen — 从零为 ncnn 新增算子的 Agent 系统

输入一个 PyTorch 模型,**从无到有**生成并验证一个 ncnn 算子:
**kernel 实现(layer)** + **计算图转换(pnnx pass)**,每一步都对 PyTorch 真值验证。

## Agent 层级
```
OperatorAgent（总控,operator_agent.py）
├── KernelAgent  (kernel_agent.py)   从零写 ncnn layer kernel(base/arm/vulkan),数值 vs PyTorch(LayerOracle)
│     └── layer_oracle/  方案A runner:编译候选 .cpp + libncnn.a,对 PyTorch allclose
├── GraphAgent   (graph_agent.py)    从零写 pnnx 计算图转换,结构 / 端到端数值验证
│     └── graph_pipeline  探针 grounding / 注入 / 编译(tree-sitter 定位)/ 转换 / 验证
└── AdapterAgent (orchestrator/adapter_agent.py)   端到端失败时,把"算法对的" kernel 改造成
      符合 ncnn Layer-Net 契约(mb.load 权重 type / forward overload 与 flag / param-id),
      喂 ncnn_interface/ncnn_contract.md(C1-C6)+ 真实 .ncnn.param + 内置参考实现
端到端:KernelAgent → 装层+重建 libncnn → GraphAgent(强制目标=新层)/ 已原生则用 baseline
        → NetOracle 跑转换模型 vs PyTorch →(失败)AdapterAgent 契约修复 → production 校验
```

## 目录结构
```
opgen/
  config.py                 路径与运行参数(GraphConfig, RUNS_ROOT, 自动定位 kernelgen/ncnn)
  llm_api.py                多 provider LLM(按 model 名路由:DeepSeek / OpenRouter;流式 + 默认关 reasoning)
  kernel/                   kernel_agent.py / kernel_pipeline.py / kernel_prompts.py / kernel_schemas.py
  graph/                    graph_agent.py  / graph_pipeline.py  / graph_prompts.py  / graph_schemas.py
  ncnn_interface/           110 层接口字典 + ncnn_contract.md(C1-C6 Layer-Net 契约);lookup.py 把 param-id/权重/flag 注入 prompt
  orchestrator/             operator_agent.py(总控) + adapter_agent.py(契约修复) + production_validation.py
  optimize/                 OptimizeAgent(两层 QD 优化器)
  layer_oracle/             LayerOracle(单层) + NetOracle(整模型) + 两个通用 C++ runner
  tools/                    文件/shell 工具
  cli/run_operator_agent.py CLI:总控(推荐)
  cli/run_kernel_agent.py   CLI:只跑 kernel
  cli/run_graph_agent.py    CLI:只跑计算图转换
  cli/run_layer_oracle.py   LayerOracle 自检(conv1d)
  docs/                     设计文档 + 各阶段验证报告
  runs/                     运行产物(gitignore)
    <task>/kernel/          KernelAgent 产物(analyzer/profile/round_XX/*.h/.cpp/result.json)
    <task>/graph/           GraphAgent 产物(pnnx_ir_probe/round_XX/*.ncnn.param ...)
    <task>/operator/        总控合并 summary.json
    _oracle/  _net/         oracle 编译/运行临时区
```

## 一次性前提
1. **Python 环境 + 依赖**(Python 3.12;依赖见仓库根 `requirements.txt`:`numpy torch openai ncnn cmake`)。两选一:
   - **conda(能装 conda 时推荐)**:
     ```bash
     conda create -n qdkernel python=3.12 -y && conda activate qdkernel
     pip install -r ../requirements.txt
     ```
   - **venv(本机无法装 conda 时的等价方案,本仓库 `.venv` 即用此)**:
     ```bash
     python3.12 -m venv ../.venv && source ../.venv/bin/activate
     pip install -r ../requirements.txt
     ```
   > `cmake`/`ncnn(pyncnn)` 是**运行时**依赖:oracle 用 `cmake` 编译候选 kernel,`pyncnn` 跑转换后的 .ncnn 模型做端到端 allclose。
2. 构建一次 pnnx 与 libncnn(自动定位 `kernelgen/ncnn`):
   ```bash
   T=../.venv/lib/python3.12/site-packages/torch
   cmake -S ../../ncnn/tools/pnnx -B ../../ncnn/tools/pnnx/build -DTorch_INSTALL_DIR=$T \
         -DPython3_EXECUTABLE=../.venv/bin/python && cmake --build ../../ncnn/tools/pnnx/build -j
   cmake -S ../../ncnn -B ../../ncnn/build_lib -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
         -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_PYTHON=OFF \
         -DCMAKE_BUILD_TYPE=Release && cmake --build ../../ncnn/build_lib -j
   ```
3. **LLM key**(按 `--model-name` 选 provider,`llm_api.py` 按名路由):
   - `deepseek-v4-pro` / `deepseek-*` → `export DEEPSEEK_API_KEY=...`
   - 其它(`z-ai/...` 等)→ `export OPENROUTER_API_KEY=...`

## 运行命令
单算子(在 `opgen/` 下,用 conda/venv 的 python):
```bash
python cli/run_operator_agent.py --task Greater --backends base,arm --model-name deepseek-v4-pro
```
批量(在仓库根,一个 runner 管所有集合;可断点续跑,结果里已有的算子跳过):
```bash
DEEPSEEK_API_KEY=... python batch/batch_runner.py --set miniset --model deepseek-v4-pro
# sets: miniset(11) / subset(~26) / all(183);--ops A,B 只跑指定算子
```
