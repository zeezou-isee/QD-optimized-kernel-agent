# 端到端"从零新增 ncnn 算子"验证报告(修正后)

- 日期:2026-06-17
- 模型:`z-ai/glm-5.1`(OpenRouter)
- 用例:从数据集 `MobileKernelBench_git/dataset/Mobilekernelbench` 中选的**两个真未实现算子**
  - `Logic/Greater`(`torch.gt`,逐元素 a>b)
  - `Logic/Less`(`torch.lt`,逐元素 a<b)
  - 二者 `baseline_supported=False`(ncnn BinaryOp 只有算术、无比较算子,pnnx 也不转)。

## 修正了什么(针对上一次的假阳性)

上一轮默认编排里,graph 阶段把 `torch.gt` 错误地映射成了 `BinaryOp 0=4`(MAX),结构检查却"通过"——
**假阳性**。两处根因 + 修正:

1. **阶段脱节** → **耦合**:`GraphAgent` 新增 `force_target_layer`,orchestrator 把 KernelAgent 写的新层名
   (`Cand_Greater`)强制设为转换目标,prompt 硬约束"`type_str()` 必须返回该层、禁止复用 BinaryOp 等已有算子"。
2. **结构验证查不出语义** → **端到端数值**:新增 `NetOracle`——把验证过的 kernel 装进 `ncnn/src/layer` +
   `ncnn_add_layer` + 重建 `libncnn.a`,再用通用 `net_oracle_runner`(链接该 libncnn)实跑转换出的
   `.ncnn.param/.bin`,对 PyTorch `allclose`。这能抓住 gt→max 这类语义错误。
   - 关键坑修复:内置层不能自带 `DEFINE_LAYER_CREATOR`(`ncnn_add_layer` 会生成),install 时自动剥离,避免 duplicate symbol。

## 编排流程(operator_agent.py)
```
Phase1 KernelAgent  → 写 kernel,数值 vs PyTorch(LayerOracle,独立编译)
Bridge              → 装 kernel 进 src/layer + ncnn_add_layer + 重建 libncnn.a
Phase2 GraphAgent   → 写 pnnx 转换,强制目标 = 新层(结构验证)
Verify              → NetOracle 跑转换模型 vs PyTorch(端到端 allclose)
Cleanup             → 还原源码树 + 重建 libncnn 干净
```

## 结果:两个算子均**真端到端通过** ✅

| 算子 | kernel 数值 | 装层+重建 | graph(目标) | 端到端数值 | 总体 |
|---|---|---|---|---|---|
| Greater (torch.gt) | ✅ max_diff=0.0 | ✅ | ✅ `Cand_Greater` | ✅ **max_diff=0.0** | **success** |
| Less (torch.lt) | ✅ max_diff=0.0 | ✅ | ✅ `Cand_Less` | ✅ **max_diff=0.0** | **success** |

转换产物确认指向新层(非复用旧算子):
```
# Greater.ncnn.param
Cand_Greater   cand_greater_0   2 1 in0 in1 out0
# Less.ncnn.param
Cand_Less      cand_less_0      2 1 in0 in1 out0
```
端到端数值:`net_oracle_runner` 加载该 `.ncnn.param/.bin`(libncnn 已含新层),喂同一输入,输出与
PyTorch `torch.gt/lt` 逐元素一致(max_diff=0.0)。

## 零侵入确认
两个算子跑完后 `git status` 干净——install 的 kernel 文件与 `ncnn_add_layer` 改动在 Cleanup 阶段被还原,
libncnn 重建回干净状态。(如需保留产物,`--keep-installed`。)

## 结论
对**真正未实现**的算子,整套系统实现了"从无到有新增一个 ncnn 算子"的完整闭环并**端到端数值验证**:
> KernelAgent 写并验对 kernel → 装入 ncnn → GraphAgent 写并验对计算图转换(强制指向新层)→
> 用真实 ncnn 运行转换模型对 PyTorch 验证。语义正确性由端到端数值保证,不再有结构假阳性。

## 复现
```bash
export OPENROUTER_API_KEY=...
cd EndtoEndMobilekernelAgent
.venv/bin/python run_operator_agent.py --task Greater --model-name z-ai/glm-5.1 \
    --torch-install-dir $PWD/.venv/lib/python3.12/site-packages/torch
.venv/bin/python run_operator_agent.py --task Less   --model-name z-ai/glm-5.1 \
    --torch-install-dir $PWD/.venv/lib/python3.12/site-packages/torch
```
