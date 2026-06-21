# 注册算子的端到端验证(原生 pnnx + ncnn,无 agent)

数据来源:`MobileKernelBench_git/dataset/Mobilekernelbench`;流程:PyTorch 模型 → 原生 `pnnx` 转换 → ncnn 运行 → 对 PyTorch `allclose(2e-3)`。

| 算子 | pnnx 转换 | 转成的 ncnn 层 | 原生支持(无残留) | 数值正确 | max_diff |
|---|---|---|---|---|---|
| Greater | ✅ | `Cand_Greater             gt_0                     2 1 in0 in1 out0` | ✅ | ✅ | 0.0 |
| LessEqual | ✅ | `Cand_LessEqual           le_0                     2 1 in0 in1 out0` | ✅ | ✅ | 0.0 |

**总判定:全部通过 ✅**