# pnnx_shortcut_report:graph agent 在难算子上的规律与提升方案

- 日期:2026-06-18
- 样本:从数据集 51 个真未支持算子里抽 **20 个**(10 常见 + 10 困难),跑 KernelAgent(数值 vs PyTorch)+ GraphAgent(强制目标新层)。
- 目的:找出 graph agent 收敛失败的**普遍规律**,提出提升方案。

## 1. 20 个算子结果(按 pnnx 表示对齐)

| 算子 | 结果 | pnnx 真实 op_types | 类型 |
|---|---|---|---|
| Equal | graph ✅ | `torch.eq` | 二元比较 |
| Not | graph ✅ | `torch.logical_not` | 一元逻辑 |
| Sinh | graph ✅ | `pnnx.Expression`(单个) | 一元数学 |
| Mod | graph ✅ | `pnnx.Expression`(单个) | 二元取模 |
| TopK | graph ✅ | `torch.topk` | 多输出 |
| Range | graph ✅ | `torch.arange` | 常量生成 |
| MaxPool_2d_dilations | graph ✅ | `F.max_pool2d` | 池化变体 |
| InstanceNormalization_1d | graph ✅ | `nn.InstanceNorm1d` | 归一化+权重 |
| DepthToSpace | graph ✅ | `nn.PixelShuffle` | 像素重排 |
| CumSum | graph ❌ | `torch.cumsum`(单个,但带轴) | 带轴扫描 |
| And | graph ❌ | `[]`(被拆解) | 二元逻辑 |
| Where | graph ❌ | `[]` | 三输入选择 |
| Cast | graph ❌ | `[]` | 类型转换 |
| Trilu_lower | graph ❌ | `[]` | 三角掩码 |
| GatherElements | graph ❌ | `[]` | 动态索引 |
| BitwiseAnd | graph ❌ | `[]` | 位运算 |
| ScatterElements | **kernel ❌** | — | 动态 scatter |
| OneHot | **kernel ❌** | `[]` | 索引→onehot |
| Det | **kernel ❌** | `aten::linalg_det` | 行列式 |
| Unique | **kernel ❌** | `pnnx.Expression+aten::unique_dim+prim::TupleConstruct` | 动态输出尺寸 |

汇总:**kernel 成功 16/20;graph 成功 9/20**。

## 2. 三条普遍规律

### 规律 A:graph 成功 ⟺ pnnx 表示是「单个可识别的高层算子」
9 个 graph 成功的算子,pnnx IR **全部**是一个干净的单节点:`torch.X` / `F.x` / `nn.X` / 单个 `pnnx.Expression`。
agent 的 `match_pattern_graph` 只需匹配**一个节点**→ 映射到新 ncnn 层,稳定收敛。
**`pnnx_op_types` 几乎是 graph 能否成功的预测器。**

### 规律 B【已更正】:op_types 为空 ≠「拆成子图」,主因是 **dtype harness bug**(已修)
> 初稿把"op_types 为空"归因为"算子被拆成低层子图"。**实测推翻了这一诊断**,真因有三类:
>
> **B1(主因,已修复)— 输入 dtype 没传对,pnnx 直接中止。**
> 复跑 And/Where/BitwiseAnd 的 pnnx,发现**根本没产出 .pnnx.param**,报错是:
> `input_shapes[0] expect [..]bool but got [..]f32` / `expect i32 but got f32`。
> 原因:`make_pt` 生成的 `inputshape` 默认 f32,而这些算子需要 **bool/int** 输入 → pnnx 在 shape 校验阶段 abort → 探针拿到空 IR → op_types=`[]`。
> 用正确 dtype 重跑后,它们其实是**单个干净算子**:
> `And→torch.logical_and`、`Where→torch.where`、`BitwiseAnd→torch.bitwise_and`(都属规律 A,应能转)。
> **修复**:`make_pt` 按 `get_inputs()` 每个张量的真实 dtype 给 `inputshape` 补后缀(`bool/i32/i64/...`)。已落地并验证。
>
> **B2 — 恒等折叠(Cast)。** Cast 到相同 dtype = no-op,pnnx 把它折叠掉,.pnnx.param 只剩 Input→Output、.ncnn.param 只有 Input。无算子可转(这是 trace/数据集产物,非缺陷)。
>
> **B3 — 真正的多节点/子图算子(存在但不是上面这些)。** pnnx 对**确实会分解**的算子用**多节点 `convert_*` pass** 处理,例如 `pass_ncnn/convert_torch_cat.cpp`、`convert_Tensor_slice.cpp`、`expand_expression.cpp`。这类才需要写多节点 `match_pattern_graph`,是真正较难的一类。
>
> 结论:之前 6 个"被拆解"失败里,**多数是 dtype bug(B1)**,修复后回归规律 A;只有真正 B3 类才是 graph 的硬骨头。

### 规律 C:单算子但需「轴/参数映射」也会失败(CumSum)
`CumSum` 的 pnnx 是干净的 `torch.cumsum`,但 graph 仍失败 —— 因为要把 `dim/axis` 正确映射到
ncnn `CumulativeSum` 的 param-id。**单节点是必要不充分**:带非平凡参数/轴的映射会拉低收敛率。

### 附:kernel 失败 ⟺ 动态输出 / 复杂算法
ScatterElements、OneHot(输出尺寸依赖 num_classes)、Det(迭代行列式)、Unique(动态输出+多输出 tuple)。
这些 ncnn 本身缺乏对应原语/动态 shape 支持,**不应让 agent 在固定轮数里硬试**。

## 3. 提升 graph agent 的方案(按杠杆排序)

### ⓪ 修 dtype harness bug(最高杠杆 + 最便宜,已完成)
`make_pt` 的 `inputshape` 按输入真实 dtype 补后缀(bool/i32/i64/...)。**一行级修复,直接救回 B1 类
(And/Where/BitwiseAnd 等所有 bool/int 输入算子)**,它们随即变成"单算子"走规律 A。这是本次最重要的发现。

### ① 让探针 IR「按失败模式分流 + 喂真实子图」(针对真正的 B3 多节点算子)
agent 已经在生成前跑 pnnx 探针。现在只在 op_types 非空时有用。改成:
- **op_types 是单算子** → 现有路径(已稳)。
- **op_types 为空 / 多节点** → 把 `.pnnx.param` 里**该算子对应的完整子图原文**(多节点 + 连接 + 属性)
  作为 grounding 直接喂给 coder,并要求写**多节点 `match_pattern_graph`**(或 `replace_pattern_graph`)。
  现在 coder 在这类算子上是"盲写",这一步把"要匹配什么"变成已知事实。

### ② 最近邻「子图匹配」范例检索(配合①)
对被拆解的算子,检索 ncnn 现有的**多节点/子图改写** pass 作范例:
`pass_ncnn/convert_Tensor_slice.cpp`、`convert_*`、`expand_expression.cpp`、表达式相关 pass。
让 coder 模仿"如何匹配 Tensor 索引/表达式子图",而不是只给它单算子范例(F_hardsigmoid 那类)。

### ③ 轴/参数 grounding(直击规律 C)
探针解析出该 pnnx 节点的属性(如 `cumsum` 的 `dim`),连同**目标 ncnn 层的 param-id 表**
(`operation-param-weight-table.md`)一起喂给 coder,把"哪个 param 填什么"变成显式信息。

### ④ kernel 可行性预筛(直击 kernel 失败)
探针发现**动态输出尺寸**(输出 shape 依赖输入值:Unique/OneHot/Scatter)或**已知复杂算子**(Det)时,
标记为"超出基础 kernel 范围"并直接报告,**不浪费迭代轮数**。可作为一个独立的 `feasibility` 阶段。

### ⑤ graph 阶段「换策略」重试 + 提高轮数
当前固定 4–5 轮、单一策略。可在连续失败时切换策略(单算子匹配 ↔ 子图/replace_pattern),
并对"被拆解"类算子给更高轮数预算。

## 3.5 dtype 修复后的重测(本次新增 · 实证)

把首批失败的 11 个算子用修复 dtype 后的 harness **完整端到端**(kernel→install→graph→Net 数值)全部重跑。

| 算子 | 之前 | 现在 | kernel | graph | e2e 数值 | max_diff |
|---|---|---|---|---|---|---|
| **And** | graph(dtype) | **✅ success** | ✅ | ✅ | ✅ | 0.0 |
| **Where** | graph(dtype) | **✅ success** | ✅ | ✅ | ✅ | 0.0 |
| **BitwiseAnd** | graph(dtype) | **✅ success** | ✅ | ✅ | ✅ | 0.0 |
| **CumSum** | graph(axis) | **✅ success** | ✅ | ✅ | ✅ | 1.4e-6 |
| **GatherElements** | graph | **✅ success** | ✅ | ✅ | ✅ | 0.0 |
| Cast | graph(identity-fold) | ❌ fail | ✅ | ❌ | — | pnnx 把 cast 折叠成 no-op |
| Trilu_lower | graph | ❌ fail | ✅ | ❌ | — | `aten::tril` 残留 |
| ScatterElements | kernel | ❌ fail | **✅(意外)** | ❌ | — | pnnx 出 `Expression + aten::scatter` |
| OneHot | kernel | ❌ fail | ❌ | — | — | 动态输出尺寸(num_classes) |
| Det | kernel | ❌ fail | ❌ | — | — | `aten::linalg_det` 算法复杂 |
| Unique | kernel | ❌ fail | ❌ | — | — | 动态输出 + 多输出 tuple |

**重测汇总:5/11 救回**(45%,首测同批 0/11),全部 e2e 数值 max_diff ≤ 1.4e-6。

**关键发现:**
- **dtype 修复直接救回 5 个**:And/Where/BitwiseAnd(`torch.logical_and`/`torch.where`/`torch.bitwise_and`)
  + CumSum(`torch.cumsum`)+ GatherElements(`torch.gather`)—— 修复后 pnnx 给的全是干净单算子,走规律 A 一次通过。
- **剩余 6 个失败的真根因分得很清**:
  - Cast(B2 折叠)+ Trilu_lower(`Expression+aten::tril`)+ ScatterElements(`Expression+aten::scatter`)
    = **真·多节点/被拆解算子(B3)** —— 需要写多节点 `match_pattern_graph`,这才是 graph agent 真正的难点。
  - OneHot / Det / Unique = **kernel 本身不可行**(动态输出 / 复杂数学 / 多输出 tuple)—— 应在可行性阶段预筛,不让 agent 硬试。
- 即:**真正的 graph 短板只占 3/11(约 27%),而非首测看起来的 ~100%**。harness 假阴性占了大头。

> 教训:首次评估覆盖率,有相当一部分"失败"其实是 **harness 假阴性**(dtype/折叠/轮数)。修一行 dtype + 增加轮数预算,
> 真实成功率显著高于初测数字。

---

## 3.6 命令式 pass 注入路径验证(本次新增 · 关键发现)

针对剩余 3 个真·多节点失败算子(Cast/Trilu/ScatterElements),源码调研 ncnn 自己怎么处理 → 实测验证命令式注入。

### 事实:ncnn 处理多节点子图用两种风格,都是 GraphRewriterPass 接口
| 风格 | 文件举例 | 写法 |
|---|---|---|
| **A 模式串(声明式)** | `pass_level2/F_hardsigmoid_2.cpp`(10 节点)、`pass_ncnn/F_logsigmoid.cpp`(`replace_pattern_graph`) | 在 `match_pattern_graph()` 返回多节点 PNNX-IR 文本,框架做子图匹配/替换 |
| **B 命令式(过程式)** | `pass_ncnn/convert_torch_cat.cpp`、`convert_Tensor_slice.cpp`、`expand_expression.cpp`(共 17 个 `convert_*`) | 写 `void convert_X(Graph&)`,直接 `for op:g.ops` 改 `op->type/params/inputs`,可插/删节点 |

为什么需要 B?**模式串 DSL 表达不了的事**:跨节点取参数、负轴归一化/广播、一拆多/多合一、表达式展开等。

### 验证实验:Trilu_lower 命令式注入
手写 `pass_ncnn/convert_cand_trilu.cpp`(命令式遍历 graph,把 `aten::tril` 改写为 `Cand_Trilu` + 顺手清掉孤立的常量 expr 节点)+ 配套 kernel。整条注入路径走通:

```
.ncnn.param 干净到位:
Input        in0      0 1 in0
Cand_Trilu   trilu_0  1 1 in0 out0 0=0 1=0

Net 加载 + 跑 → 对 PyTorch max_diff=0(整数全等)
```
**结论:命令式路径完全可行。** ncnn DSL 不是表达力不够,而是 agent 没用到这条路径。

### 为什么当前 agent 写不出 Trilu/Cast/ScatterElements
不是 ncnn 能力问题,而是 agent 的 3 处缺失:
1. **只走模式串路径**:从没见过命令式范例,也不知道有这个选项;
2. **inject_files 不支持命令式注入**:命令式比模式串**多 2 件事**——需要写 `.h` 头文件,需要 patch `pass_ncnn.cpp`(`#include + 在 pass_ncnn() 函数体加调用`);当前 `inject_files` 两件都不做,即便 LLM 写出命令式 pass 也注不上去;
3. **prompt 没教**两种写法的取舍 + 命令式范式(`while(true){...break;}` 防迭代失效、清理孤立上游节点、命名空间 `pnnx::ncnn` 等)。

### 据此的修正改进方案(取代/补强第 3 节的 ①②)
**P1.** `graph_pipeline.inject_files` 扩 2 个能力:接收 `pass_ncnn/convert_*.h`、安全 patch `pass_ncnn.cpp` 的两个锚点(`#include` 区 + `pass_ncnn(Graph&...)` 函数体)。
**P2.** `graph_prompts.coder_prompt` 增加"两种写法可选"指引 + 命令式范例(`convert_torch_cat.cpp` 一份)+ 范式注意事项(while/break、清理孤立节点等)。
**P3.** 范例检索按"目标算子复杂度"分流:`op_types` 含 `Expression/aten::xxx` 残留 → 优先拉命令式 `convert_*` 范例;否则拉模式串范例。

> 这是底层能力齐全、agent 没接上的典型工程问题,不是模型能力问题。

### 改进后实测(P1+P2+P3 落地后重跑 3 个真·多节点失败算子)
| 算子 | 此前 | 改进后 | 说明 |
|---|---|---|---|
| **Cast** | graph(B2 fold) | **✅ success**(e2e diff=1.9e-3) | 救回 |
| Trilu_lower | graph(Expression+aten::tril) | ❌ graph (structural) | LLM 同时写了 pass_level2+pass_ncnn+命令式 5 个文件,但目标 op_type 名错位,IR 仍残留 `aten::tril` |
| ScatterElements | kernel | ❌ kernel (numeric)(max_diff=0.998) | 这次能编译,但 LLM 没写对 scatter 算法 |

**改进结论:**
- P1/P2/P3 工具/prompt 改造**机制正确**(Cast 端到端跑通验证)。
- 剩余两个不是"路径不通",是**模型自身**写代码的问题:
  - Trilu:LLM "贪多 / 文件混搭",同时上模式串和命令式,且名字写错。下一步可在 prompt 里**强制单一路径**(IR 残留 => 只写命令式 convert_*,禁写模式串)。
  - ScatterElements:算法本身复杂,kernel 写不对 — 仍属 kernel 可行性问题(规律 C+/④)。

---

## 4. 结论(更正后)
- **kernel 写作不是瓶颈**(16/20)。
- 初测 graph 9/20,但**复盘发现失败里很大一部分不是 agent 的能力问题,而是 harness 的 dtype bug**:
  bool/int 输入算子(And/Where/BitwiseAnd 等)被传成 f32,pnnx 直接中止 → 误判为"被拆解/不可转"。
  **修复 dtype 后,这些其实是单算子,应回归可转**(规律 A)。
- 真实的 graph 难点收敛为三类:
  1. **真正多节点/子图算子**(B3,如 cat/slice/expression 类)——需要写多节点 pattern,靠①②改进;
  2. **带轴/参数映射**(CumSum)——靠③ grounding;
  3. **kernel 本身不可行**(动态输出 Unique/OneHot/Scatter、复杂数学 Det)——靠④预筛,不该硬试。
- 最高杠杆且已完成的改进就是 **⓪ 修 dtype bug**;其余①②③④进一步提升真·难算子覆盖,且都不需改 agent 主架构。

> 教训:评估覆盖率前必须先排除 harness 假阴性(dtype/折叠)。本次"op_types 为空"= 探针失败信号,
> 不能直接当作"算子复杂"。

## 5. 数据
- `batch_results.json`(常见 10)、`batch_hard_results.json`(困难 10)、`probe_classify.json`(全 51 分类)。
