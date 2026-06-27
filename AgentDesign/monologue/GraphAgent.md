# GraphAgent — PyTorch→ncnn 计算图转换 agent 设计

## TL;DR

GraphAgent 解决 KernelAgent 解决不了的另一半：**一个 ncnn 还不认识的算子，光有 kernel 跑不起来——必须先有 PyTorch→ncnn 的图转换（PNNX pass）**。它用"状态机 agent loop + 三角色（analyzer/coder/debugger）+ 5 阶段函数式 pipeline"写出并验证这套 pass。

- **何时跑**：在编排器里是**有条件**的——KernelAgent 之后、存在性检查 [3] 判定"ncnn 不能原生转换"时才进 [3b]；已支持则跳过（用原生转换 + `.param` 定向）。
- **写什么**：`pass_level2/F_<op>.cpp`（aten/prim → PNNX `F.x`）+ `pass_ncnn/F_<op>.cpp` 或 `convert_<op>.cpp/.h`（PNNX op → ncnn layer）+ 端到端测试。
- **两种风格**：**pattern**（声明式 `match_pattern_graph`，1:1 干净映射）与 **imperative**（过程式 `convert_<op>(Graph&)`，处理多节点子图）。
- **怎么验**：5 阶段——inject → build pnnx → convert（跑出 `.ncnn.param`）→ structural（目标层在不在）→ numeric（整网 allclose vs PyTorch）；每轮只修第一个失败阶段。
- **关键加固**：analyzer 以 **baseline pnnx 探测**为 ground truth 定 `target_ncnn_layer`，避免把 `torch.log` 误判成不存在的 "Log" 层（实为 `UnaryOp`）——软（prompt）+硬（机械纠正）两层。

---

## 1. 它解决什么（与 KernelAgent 的分工）

- **KernelAgent** 写算子的**数学实现**（`.cpp`），独立验证（LayerOracle 对拍 PyTorch），不需要图。
- **GraphAgent** 写算子的**图转换**：让 PNNX（`ncnn/tools/pnnx`）能把 PyTorch 里的这个 op 转成一张 ncnn 能加载的 `.ncnn.param`/`.bin`。没有它，一个 ncnn 不认识的新 op 根本无法被装进模型跑起来。

这正是经典 MoKA 类工作缺的一环——它们只能改写"框架里已存在算子"的 kernel，无法引入新算子。

## 2. 在整体 pipeline 中的位置（条件触发）

```
[1] KernelAgent 写并验证 kernel（Cand_<Op>）
[2] Bridge 装进 libncnn
[3] 存在性检查 probe_pnnx_ir：ncnn 已能转换此 op?
      ├─ 是 → 用原生转换，跳过 GraphAgent（再 .param 定向到 Cand_<Op>）
      └─ 否 → [3b] GraphAgent（force_target_layer=Cand_<Op>）写 PNNX pass
[4] 整网数值 → [5] production → [6] QD 优化
```

所以 GraphAgent 不是每次都跑；详见 [OperatorPipeline.md](./OperatorPipeline.md)。

## 3. PNNX 两阶段转换（领域知识）

一条转换分两段（`graph_prompts.py: PNNX_BACKGROUND`）：

1. **torch → pnnx**
   - `pass_level1/nn_Xxx.cpp`（`FuseModulePass`）——仅当 op 是 `nn.Module` 时；用 `match_type_str()` 匹配 python 模块路径。**多数情况不需要**（nn.* 已被捕获）。
   - `pass_level2/F_xxx.cpp`（`GraphRewriterPass`）——把 `aten::*`/`prim::` 子图映射成 PNNX op `F.xxx`。一个 torch op 可能展开成多个子图，**常需多个 match-pattern 变体**。
2. **pnnx → ncnn**
   - `pass_ncnn/F_xxx.cpp`（`GraphRewriterPass`，`namespace pnnx::ncnn`）——把 PNNX op 重写成 ncnn layer，填 params/weights。`type_str()` 返回的 ncnn 层名**必须是真实存在的 ncnn layer**。

注册宏：`REGISTER_GLOBAL_PNNX_*`（level1/level2/ncnn 各一）。头文件/基类必须精确（`fuse_module_pass.h` / `pass_level2.h` / `pass_ncnn.h`）——这是 LLM 高频写错处。

## 4. 两种 pass 写法（按场景选）

**A) pattern 风格（声明式，默认）**：子类 `GraphRewriterPass` + `match_pattern_graph()`。适用于源 IR 是**单个高层 op**（`F.x`/`nn.X`/`torch.x`）且 **1:1 干净映射**到某个 ncnn layer。

**B) imperative 风格（过程式）**：写 `void convert_<op>(Graph& g)` 自己遍历图。**当遇到以下情况用它**：
- 源 op 分解成**多节点子图**（如 `pnnx.Expression + aten::xxx`）；
- 参数要从**多个上游节点**（常量、shape）算；
- 需要负轴/batch 归一化，或一个 op 拆成多个；
- pattern DSL 表达不了。

> **多节点子图就是靠 B 解决的**——runs 里 `Det_3d`（`convert_F_det_3d.cpp/.h`）、`Greater`（`convert_log.cpp/.h`）都是 imperative 路径并成功。

imperative 已知暗坑（prompt 里列出）：必须同时写 `.h`；**不要**手写注册宏（harness 自动接进 dispatcher）；吸收上游常量后必须删 orphan `pnnx.Expression`，否则 `.ncnn.param` 漏出 `layer pnnx.Expression not exists`；`while(true){match-one; break}` 因为改图会让迭代器失效。

## 5. Agent loop：状态机 + 三角色 + 5 阶段 pipeline

与 KernelAgent 同构（`graph_agent.py`）：

**5 阶段函数式 pipeline**（`graph_pipeline.py`，`GraphResult` 逐阶段记录）：
```
inject     写 pass 文件 + 改两个 CMakeLists（带备份；imperative 自动接 dispatcher）
build      cmake 增量编 pnnx
convert    跑 pnnx → .pnnx.param / .ncnn.param / .bin
structural verify_structural：目标 ncnn 层在不在 .ncnn.param 里、op 匹配没
numeric    verify_numeric：ctest 跑端到端 allclose vs PyTorch
```

**循环**：每轮只修"第一个失败的阶段"（`first_failure()`：inject→build→convert→structural→numeric），把该阶段的诊断喂给 debugger 角色；≤`graph_max_rounds`（默认 15）。runs 里看到的 phase 如 `identify_and_generate` / `convert_repair` 即对应。

**三角色**：analyzer（定 OpProfile：source_form/category/target_ncnn_layer/files_to_write）、coder（首轮产出 pass 文件）、debugger（按失败阶段修）。

## 6. Grounding：用 pnnx 探测做 ground truth（关键加固）

**问题**：analyzer 若凭 LLM 猜 `target_ncnn_layer`，会把折叠进通用 op 的算子误判成不存在的同名层——`torch.log` 实际折叠成 `UnaryOp`，却被判成 "Log"，结构检查必挂（runs `Log` 因此跑满 15 轮失败：`Target 'Cand_Log' NOT found ... Present: ['Input','UnaryOp']` + 反复写错 `write()` override 签名）。

**修复（`graph_agent.py` + `graph_prompts.py`）——软+硬两层**：
- **软**：`run()` 先 `probe_pnnx_ir` 探测真实 IR（`op_types` / `residual_aten` / baseline `.ncnn.param`），把 `grounding` 喂给 `analyze()` → `analyzer_prompt`，并加 `_TARGET_JUDGMENT` 规则：从 IR 决定、别猜；`log/exp/sqrt→UnaryOp`、`gt/add→BinaryOp`、`sum→Reduction`；只有存在未转换 aten 残留才新建 `Cand_<Op>`；绝不发明 IR 里没有的层名。
- **硬**：`_ground_target()` 机械护栏——对"完全原生可转换"（无 aten 残留）的算子，若目标层不在真实 `op_types` 里，直接纠正到真实计算层并标 `already_supported=True`。**只在完全原生情形触发**，真新算子（有残留）绝不误改。

效果（单测）：`Log`→纠正为 `UnaryOp`/supported；`Greater`（有 `aten::gt`）→保持 `Cand_Greater` 不动；`Exp`（已对）→标 supported。

## 7. 编排器集成开关

- **`force_target_layer`**：编排器对新算子传 `Cand_<Op>`，让转换直接产出我们的层名（analyze 之后覆盖 `target_ncnn_layer`）。
- **`skip_if_supported`**：探测发现已原生支持就停（编排器 [3] 用这条 skip 掉 GraphAgent）。
- 注意：`baseline_supported` 仅信息性——验证始终以 **PyTorch 为 oracle**（`verify_numeric = allclose(torch_out, ncnn_out)`），不依赖任何 baseline。

## 8. 能力与边界（runs 实证）

**稳**（1–2 轮成功）：能干净映射到某个 ncnn layer 的 op——`Greater`(gt)、`MatMul` 各变体、`Det_3d`、`ReduceSum`、`Gemm`；多节点子图也能经 imperative 跑通。

**不稳/难**（见 `Log` 等）：
1. **折叠进通用 op**（UnaryOp/BinaryOp/Reduction/Expression）的 op——target 易判错（**已由 §6 修复**）。
2. **pnnx C++ API 不宽容**——`GraphRewriterPass::write()` 的重载/override 签名（2/3 参 vs 写成 4 参标 override）反复编译错，难自愈。
3. **覆盖已原生支持的 op**——强写自定义 pass 与原生转换冲突（靠 [3] 存在性检查规避，而非 GraphAgent 解决）。
4. **重度 imperative 图改写**——Expression 清理、operand/consumers 维护、轴归一化等暗坑多。

## 附：关键代码位置

| 主题 | 位置 |
|---|---|
| Agent loop / 角色 / probe→analyze→循环 | `opgen/graph/graph_agent.py: run / analyze / _ground_target / _run_pipeline` |
| 5 阶段 pipeline + 探测 + 注入 | `opgen/graph/graph_pipeline.py: probe_pnnx_ir / inject_files / build_pnnx / run_conversion / verify_structural / verify_numeric` |
| 领域知识 + 三角色 prompt + grounding 注入 | `opgen/graph/graph_prompts.py: PNNX_BACKGROUND / analyzer_prompt / _TARGET_JUDGMENT / coder_prompt / debugger_prompt / format_grounding` |
| OpProfile / GraphResult（含 already_supported、stages） | `opgen/graph/graph_schemas.py` |
| 编排器集成（force_target / skip / 存在性检查） | `opgen/orchestrator/operator_agent.py: _check_already_in_ncnn` |
