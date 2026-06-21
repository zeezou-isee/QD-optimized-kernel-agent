# 10 类未支持算子的 agent 端到端测试

对每个真未支持算子跑:kernel 数值(vs PyTorch) + 计算图转换(强制目标新层) + 端到端数值(Net 跑转换模型 vs PyTorch)。临时验证,跑完还原源码树。

| 算子 | 种类 | kernel | graph | 端到端数值 | 总判定 |
|---|---|---|---|---|---|
| Equal | 二元比较 | success | success | ✅ 0.0 | ✅ success |
| And | 二元逻辑 | success | fail | —  | ❌ fail |
| Not | 一元逻辑 | success | success | ✅ 0.0 | ✅ success |
| Sinh | 一元数学(表达式) | success | success | ✅ 9.5367431640625e-07 | ✅ success |
| Where | 三输入选择 | success | fail | —  | ❌ fail |
| Cast | 类型转换 | success | fail | —  | ❌ fail |
| Mod | 二元取模 | success | success | ✅ 0.0 | ✅ success |
| CumSum | 带轴扫描 | success | fail | —  | ❌ fail |
| Trilu_lower | 三角掩码 | success | fail | —  | ❌ fail |
| TopK | 多输出(难) | success | success | ✅ 0.0 | ✅ success |

**通过(完整端到端成功):5/10**