# agents — 从零为 ncnn 新增算子的 Agent 系统

输入一个 PyTorch 模型,**从无到有**生成并验证一个 ncnn 算子:
**kernel 实现(layer)** + **计算图转换(pnnx pass)**,每一步都对 PyTorch 真值验证。

## 四层 Agent
```
OperatorAgent（总控,orchestrator/operator_agent.py）
├── KernelAgent   (kernel/)        从零写 ncnn layer kernel(base + arm),数值 vs PyTorch(LayerOracle)
├── GraphAgent    (graph/)         从零写 pnnx 计算图转换,结构 / 端到端数值验证
└── OptimizeAgent (optimize/)      两层 Quality-Diversity 优化器:让已验证的 kernel 更快
端到端:KernelAgent → 装层+重建 libncnn → GraphAgent(强制目标=新层)→ NetOracle 跑转换模型 vs PyTorch
```

三个子 agent 同构:**agent loop(状态机) + 功能函数 + 三角色(analyzer / coder / debugger)**,
loop 每轮修复**第一个失败阶段**,只把该阶段的诊断喂给角色。

## 目录结构
```
agents/
  __init__.py               包初始化 + bootstrap_paths()(把各子目录加进 sys.path,支持扁平 import)
  config.py                 路径与运行参数(GraphConfig;自动定位 frameworks/ncnn 与 datasets/MobileKernelBench)
  llm_api.py                OpenRouter LLM(流式 + 默认关 reasoning)
  kernel/                   KernelAgent:kernel_agent / kernel_pipeline / kernel_prompts / kernel_schemas
  graph/                    GraphAgent:graph_agent / graph_pipeline / graph_prompts / graph_schemas
  orchestrator/             OperatorAgent 总控 + production_validation
  optimize/                 OptimizeAgent(QD):schemas + evaluator/ + inner/ + policy/ + proposer/ + test_m1/2/3
  layer_oracle/             LayerOracle(单层) + NetOracle(整模型) + 两个通用 C++ runner
  tools/                    文件 / shell 工具
  cli/                      CLI 入口:run_kernel_agent / run_graph_agent / run_operator_agent /
                            run_layer_oracle / run_arm_batch
  docs/                     设计文档 + 各阶段验证报告
  runs/                     运行产物(gitignore):<task>/{kernel,graph,operator},_oracle/ _net/
```

## 一次性前提
1. venv + 依赖:`torch numpy openai cmake ncnn tree_sitter tree_sitter_cpp pyyaml`。
2. 构建一次 pnnx 与 libncnn(自动定位 `frameworks/ncnn`):
   ```bash
   T=$(python -c "import os,torch;print(os.path.dirname(torch.__file__))")
   cmake -S ../frameworks/ncnn/tools/pnnx -B ../frameworks/ncnn/tools/pnnx/build \
         -DTorch_INSTALL_DIR=$T -DPython3_EXECUTABLE=$(which python) && \
     cmake --build ../frameworks/ncnn/tools/pnnx/build -j
   cmake -S ../frameworks/ncnn -B ../frameworks/ncnn/build_lib \
         -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF \
         -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release && \
     cmake --build ../frameworks/ncnn/build_lib -j
   ```
   > oracle 会优先用 `frameworks/ncnn/build_lib/`,没有则回退到已存在的 `build/`。
3. `export OPENROUTER_API_KEY=...`

## 运行命令
在**仓库根目录**下运行(脚本会自己把仓库根与 `agents/` 加到 `sys.path`):
```bash
# 写 + 验证 kernel(对 PyTorch allclose)
python agents/cli/run_kernel_agent.py --task Exp --backend base --model-name z-ai/glm-5.2
python agents/cli/run_kernel_agent.py --task Exp --backend arm  --model-name z-ai/glm-5.2   # 需先有 base

# 优化 kernel(两层 QD;真机实测)
python agents/optimize/run_optimize.py --task Exp --backend arm --policy map_elites --map-budget 20

# 端到端“新增一个 ncnn 算子”(kernel + graph + 验证 [+ 优化])
python agents/cli/run_operator_agent.py --task Greater --backends base,arm --optimize

# 只跑计算图转换 / LayerOracle 自检
python agents/cli/run_graph_agent.py --task HardSigmoid
python agents/cli/run_layer_oracle.py

# 单元测试(不需要 LLM / ncnn)
python agents/optimize/test_m1.py && python agents/optimize/test_m2.py && python agents/optimize/test_m3.py
```

## 作为库调用
```python
import agents; agents.bootstrap_paths()      # 让扁平 import 生效
from operator_agent import OperatorAgent
from config import GraphConfig
summary = OperatorAgent(task_name="Greater", cfg=GraphConfig()).run()
```

> 设计原理见 [`docs/`](docs/):`ncnn_graph_conversion.md`(背景)、`graph_agent_design.md`(方案)、
> `ncnn_layer_impl_and_verification.md`(kernel/oracle),以及各 `*_REPORT.md` 验证报告。
