# opgen — 从零为 ncnn 新增算子的 Agent 系统

输入一个 PyTorch 模型,**从无到有**生成并验证一个 ncnn 算子:
**kernel 实现(layer)** + **计算图转换(pnnx pass)**,每一步都对 PyTorch 真值验证。

## 三层 Agent
```
OperatorAgent（总控,operator_agent.py）
├── KernelAgent  (kernel_agent.py)   从零写 ncnn layer kernel,数值 vs PyTorch(LayerOracle)
│     └── layer_oracle/  方案A runner:编译候选 .cpp + libncnn.a,对 PyTorch allclose
└── GraphAgent   (graph_agent.py)    从零写 pnnx 计算图转换,结构 / 端到端数值验证
      └── graph_pipeline  探针 grounding / 注入 / 编译(tree-sitter 定位)/ 转换 / 验证
端到端:KernelAgent → 装层+重建 libncnn → GraphAgent(强制目标=新层)→ NetOracle 跑转换模型 vs PyTorch
```

## 目录结构
```
opgen/
  config.py                 路径与运行参数(GraphConfig, RUNS_ROOT, 自动定位 kernelgen/ncnn)
  llm_api.py                OpenRouter LLM(流式 + 默认关 reasoning)
  kernel_agent.py / kernel_pipeline.py / kernel_prompts.py / kernel_schemas.py
  graph_agent.py  / graph_pipeline.py  / graph_prompts.py  / graph_schemas.py
  operator_agent.py         总控编排(耦合两阶段 + 端到端数值)
  layer_oracle/             LayerOracle(单层) + NetOracle(整模型) + 两个通用 C++ runner
  tools/                    文件/shell 工具
  run_operator_agent.py     CLI:总控(推荐)
  run_kernel_agent.py       CLI:只跑 kernel
  run_graph_agent.py        CLI:只跑计算图转换
  run_layer_oracle.py       LayerOracle 自检(conv1d)
  docs/                     设计文档 + 各阶段验证报告
  runs/                     运行产物(gitignore)
    <task>/kernel/          KernelAgent 产物(analyzer/profile/round_XX/*.h/.cpp/result.json)
    <task>/graph/           GraphAgent 产物(pnnx_ir_probe/round_XX/*.ncnn.param ...)
    <task>/operator/        总控合并 summary.json
    _oracle/  _net/         oracle 编译/运行临时区
```

## 一次性前提
1. venv + 依赖:`torch numpy openai cmake ncnn tree_sitter tree_sitter_cpp pyyaml`(本仓库 `.venv` 已装好)。
2. 构建一次 pnnx 与 libncnn(自动定位 `kernelgen/ncnn`):
   ```bash
   T=../.venv/lib/python3.12/site-packages/torch
   cmake -S ../../ncnn/tools/pnnx -B ../../ncnn/tools/pnnx/build -DTorch_INSTALL_DIR=$T \
         -DPython3_EXECUTABLE=../.venv/bin/python && cmake --build ../../ncnn/tools/pnnx/build -j
   cmake -S ../../ncnn -B ../../ncnn/build_lib -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
         -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_PYTHON=OFF \
         -DCMAKE_BUILD_TYPE=Release && cmake --build ../../ncnn/build_lib -j
   ```
3. `export OPENROUTER_API_KEY=...`

## 运行命令(完整流程见下一条回复)
所有命令在 `opgen/` 目录下、用本仓库 venv 的 python 运行。
