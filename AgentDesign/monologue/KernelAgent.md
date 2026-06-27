# KernelAgent — 从零写 ncnn 算子 kernel 的 feedback agent 设计

## TL;DR

KernelAgent 写算子的**数学实现**（一个 ncnn `Layer` 的 `.cpp/.h`），用 LayerOracle **隔离实例化 + 对拍 PyTorch** 验证，全程不碰 ncnn 源码树。它本质就是一个**反馈修复循环**——成败押在你说的两件事：**反馈怎么设计** 和 **初始信息给多少**（后者它下了重注）。

- **循环**：3 态状态机 `generate → compile → numeric`，每轮只修第一个失败的阶段、只喂那个阶段的诊断。
- **反馈设计**：编译错 tree-sitter 定位行；数值错走**失败分类 taxonomy**（E3 形状/E4 转置/E5 仿射/E6 数值-或-不稳定）给**带标签+定位+修复指令**的反馈；外加每阶段病因提示 + 滚动记忆。
- **初始信息**：真实模型自省（shapes/state_dict）+ 领域知识 + 最近邻 ncnn 层范例 + **把易错的 ncnn param-id / 权重顺序机械算好直接喂给**（不让 LLM 猜）。
- **三后端**：base（可移植 CPU）/ arm（NEON+NC4HW4 子类）/ vulkan（GPU 子类 + `.comp`）。arm/vulkan 都子类化已验证的 base。
- **隔离保证**：直接 `new Cand_<Op>...`，不走注册表；arm 有"非退化"静态门，vulkan 有 `support_vulkan` 运行时断言。

---

## 1. 它做什么（与 GraphAgent 的分工）

- **KernelAgent**：写算子的**实现**（forward 算法），独立验证，**不需要图**。
- **GraphAgent**：写算子的**图转换**（PNNX pass），让 ncnn 能加载它。见 [GraphAgent.md](./GraphAgent.md)。

在编排器里 KernelAgent 是 [1]/[1b]，**无条件先跑**；图是否要自己写由其后的存在性检查决定。见 [OperatorPipeline.md](./OperatorPipeline.md)。

## 2. 反馈循环骨架（确实简单）

`kernel_agent.py: run()`：
```
round 0 : analyzer 定 profile（base）/ 从 base 派生（arm/vulkan）
          → coder_prompt（喂全部初始信息）→ LLM → 抽代码 → verify_kernel
round 1+: phase = result.first_failure()      # generate → compile → numeric，谁先挂修谁
          feedback = result.feedback(phase)     # 只喂该阶段的诊断
          → debugger_prompt(phase, code, feedback, memory) → LLM → verify
          result.ok ? break : 继续（≤max_rounds，默认 8）
```

验证（`kernel_pipeline.verify_kernel` + `LayerOracle`）：写候选文件 → 编译候选 `.cpp` + libncnn.a → `new Cand_<Op>()` 直接实例化跑 forward → allclose vs PyTorch。3 个阶段对应 3 个修复相位。

## 3. 决定成败之一：反馈怎么设计

**(a) 分阶段、只给当前失败阶段的诊断**（`KernelResult.feedback`）：
| 相位 | 反馈 |
|---|---|
| `generate_repair` | "没抽到合法 .h/.cpp 代码块" |
| `compile_repair` | **tree-sitter 定位过的编译错误**（`locate_build_errors`，裁到候选文件相关行）|
| `numeric_repair` | **失败分类 taxonomy 的标签 + 定位 + 修复指令**（见 §5.1）；崩溃则给 run_log 尾 12 行；隔离 guard 命中给专属消息 |

**(b) 叠加该阶段"常见病因"提示**（`debugger_prompt`）：numeric→Mat 索引用 `channel(q)`、轴/shape、权重布局；arm→elempack；vulkan→support_vulkan/1D workgroup…

**(c) 滚动记忆**（`_format_memory`，最近 4 轮）：phase + stages + feedback 摘要，防反复犯同一错。

## 4. 决定成败之二：初始信息给多少（下重注，不靠 LLM 猜）

1. **真实模型自省 `introspect_model`**：`input_shapes`、`state_dict`（key→shape）、`init_inputs`——真跑出来的 ground truth。
2. **领域知识 `NCNN_LAYER_BACKGROUND`**：Mat 布局（`channel(q)`/`cstep`）、按 `(one_blob_only, support_inplace)` 选 forward 接口、torch→ncnn 输入布局映射、权重按 `weight_keys` 顺序 load。
3. **最近邻 ncnn 层范例 `retrieve_layer_example`**：按 `analog_layer` 把现成 ncnn 层 `.h/.cpp` 整段当模板（arm/vulkan 附带 base 作父类上下文）。
4. **把易错部分机械算好再喂**（关键）：
   - `_infer_params`：从 state_dict 形状**机械计算** ncnn param-id（innerproduct/conv/layernorm/embed/scale…），**覆盖** LLM 猜测。
   - `_validate_weight_keys`：把 LLM 幻觉的权重 key 模糊匹配回真实 state_dict 并纠正。

> 设计哲学：不相信 LLM 能可靠把 PyTorch 语义映射到 ncnn param-id/权重布局，于是把这些**算死当初始信息给定**，让 LLM 只负责真正需要它的"写 forward 逻辑"。

## 5. 新增改动

### 5.1 失败分类 taxonomy（诊断驱动的反馈）

把数值失败从"一个标量 max_diff"升级成**规则化、确定、互斥**的分类（`failure_taxonomy.classify_failure`，纯 numpy）：

| 类 | 判据 | 反馈 |
|---|---|---|
| `E6_NUMERICAL_INSTABILITY` | out 含 NaN/Inf | 溢出/除零/未写全输出 |
| `E3_SHAPE_WRONG_COUNT` | `out.size≠ref.size` | 元素数比、少/多轴或 reduce 错，重推输出 shape |
| `E4_LAYOUT_PERMUTED` | 同 size、某轴置换匹配 | "输出是参考的转置,置换 (1,0)"——**直给答案** |
| `E5_VALUE_AFFINE` | `out≈a·ref+b`/-ref | 缩放/符号/偏移 → 指向 param/激活/eps |
| `E6_VALUE_NUMERICAL` | 无简单关系 | 逐通道 + top-k 错元素 `idx: got vs expected` 定位 |

接线：`oracle.verify` / `vulkan_oracle.verify` 失败时调它，`detail` 变 `[E4_…] …`；`OracleResult.failure_category` / `KernelResult.failure_category` 记录类别（落进每轮 result.json → 可做"失败类别分布/消融"统计）；反馈顺现有 `numeric_log` 流回 LLM，**未改 loop/prompt 管线**。三后端共享。

针对前面 runs 里复杂算子的失败（`Det_2d` 形状塌成 (1,1,1)、`Einsum_*` 元素数错、`Det` inf、`MatMul_square` 值乱）——已用真实失败签名构造数组离线验证：每例都从"标量/一句 shape mismatch"升级为带定位的反馈（`test_failure_taxonomy.py`，无需 LLM/torch/ncnn）。

### 5.2 arm 非退化静态门

arm 子类的 `forward(Mat&)` 与 base 同签名，若忘 override 会**静默回落到继承的 base CPU forward**，对 elementwise 仍过 allclose（假阳性）。`arm_forward_overridden()` 静态检查生成代码必须定义 `<arm_class>::forward[_inplace]`，否则即使 numeric 过也翻成失败、喂修复循环；arm prompt 也声明 override 为强制项。vulkan 因 forward 签名不同 + `support_vulkan` 断言，天然不可退化。

### 5.3 vulkan 生成通路

`backend=="vulkan"` → 用 `VulkanLayerOracle`（隔离实例化、GPU 上跑、`.comp` 运行时 `compile_spirv_module` 编译）；profile 由 base `as_backend("vulkan")` 派生（含 `.comp` 文件名）；`VULKAN_LAYER_BACKGROUND` prompt 教 LLM 写三件套；`verify_kernel` 按 backend 路由（传 `shader=`），无 GPU 时 `OracleResult.skipped` → 记 `numeric_skipped`。详见 [vulkan-verification-harness.md](./vulkan-verification-harness.md)。

## 6. 三后端验证一览

| backend | 验证方式 | 隔离/非退化保证 |
|---|---|---|
| base | LayerOracle：编译候选 + libncnn.a，`new Cand_<Op>()` 跑 CPU forward，对拍 PyTorch | 直接实例化,不走注册表 |
| arm | 同上 + 编入 base `.cpp`、`-I src/layer/arm`、`--packing 4`(NC4HW4) | `arm_forward_overridden` 静态门 |
| vulkan | VulkanLayerOracle：`find_package(ncnn)` 链接 vulkan libncnn，GPU forward(VkMat) | forward 签名不同 + `support_vulkan` 断言 |

## 7. 能力与边界（runs 实证）

- **稳（多 1 轮过）**：elementwise/unary/activation/logic——布局无关、forward 逻辑短、param 简单。
- **难（numeric_repair 反复失败）**：conv/matmul/einsum/det/reduction/多轴 tensor——布局强相关 + 算法非平凡。三类典型失败：**形状错（E3）**（轴→(w,h,c) 映射错）、**数值/算法错（E6,含 inf）**、**运行崩溃**（越界/权重 load 错）。§5.1 的 taxonomy 正是为缩小这类失败的反馈盲区而加。

## 8. 反馈/验证增强（已实现）

> 针对 arm/vulkan 与复杂算子反馈盲区的三项增强，均已落地并离线验证（`test_failure_taxonomy.py`，无需 LLM/torch/ncnn）。

### 8.1 加速后端用 base 做"算法 oracle" + 后端专属反馈

现在 arm/vulkan 都直接对拍 PyTorch。改为**先验 base 对 PyTorch、再验 arm/vulkan 对 base**——base 即算法 ground truth（且 arm 本就把 base `.cpp` 编进来了）。失败即可干净归因，并据此给后端专属反馈：

- **arm 差分定位**：若 `base==PyTorch` 但 `arm!=base` → "**NEON 路径偏离了自己已验证的 base（算法没错，错在向量化），偏离在位置 X**"——把"转写 bug"与"算法 bug"彻底分开。
- **arm lane/tail 周期性误差**：误差集中在 `i%elempack` 某 lane 或不满一个向量的尾部 → "检查 `i+4<=size` 边界与标量余数（很可能忘了 `*elempack`）"。
- **vulkan passthrough/覆盖检测**（给诊断器传 **input**）：若某段 `out==input`（而非 `==ref`）→ "**DISPATCH 覆盖不全：只处理了 N/M，其余是未改的输入 → workgroup 维度 vs dispatch 不符，用 `set_optimal_local_size_xyz(subgroup_size,1,1)`**"。
差分框架取轻量实现：base 在 [1] 已验证 == PyTorch，故 arm/vulkan 数值失败**必是端口 bug**——`verify_kernel` 直接前置 `PORT BUG:` 框架语（无需重跑 base，因 out-vs-ref 定位等价于 out-vs-base）。

实现：`failure_taxonomy.py: classify_failure / _coverage / _lane_tail`（新增 `input`/`backend` 参数 + `E8_DISPATCH_COVERAGE`）；`oracle.verify` / `vulkan_oracle.verify` 透传 `input`+`backend`；`verify_kernel` 加 PORT BUG 框架。E8 对 `f(x)=x` 部分定义域算子(abs 正数)**不误报**(同时要求 `≠ref`)，已单测。

> vulkan 的"分离 shader 编译面"与"fp16 near-miss"作为进一步细化留 TODO（§9）。

### 8.2 多 shape 验证

数值通过后，对**无权重算子**（输入维不被权重绑定）再验**同 rank 的尺寸变体**——变体由原输入**切片**得到（保证落在算子合法定义域；随机张量会破坏 log/sqrt/det）；任一变体失败即翻为失败并提示"索引硬编码到一个 shape"。catches "过了这个 shape、换个 shape 就错"。实现：`verify_kernel` 调 `_multishape_check` / `_size_variants`（无权重 + `run_numeric` 时；变体被算子拒绝则跳过）。

### 8.3 输出形状契约（前置初始信息）

`introspect_model` 现在跑一遍 forward 记录 `output_shape` / `ncnn_output_shape`（batch dropped）；`_introspect` 把它作为**显式契约**写进 coder/debugger prompt（"forward 必须分配 top 为这个 shape/元素数"）。这是**主动**前置（§5.1 的 E3 是**被动**事后检查），便宜直接降低形状类失败。

## 9. TODO（未来方向）

- **(4) kernel 作者经验池（兵器谱 for kernel）**：按 `analog_layer/category` 索引、从历史成功 run 提炼的作者经验（惯用法 + 常见坑），首轮注入。抬升 conv/matmul/reduction 等**难类别**成功率的结构性办法，而非靠更多反馈轮。
- **(5) 难类别首轮 breadth（K 候选）**：对 hard category，round 0 并行生成 K 份，留能编过/最接近的再深修（与 QD 采样精神一致）。
- **(6) 已知难算法给参考骨架**：Det(LU 分解)/gemm/reduction 等，初始信息里直接给正确算法骨架 + 数值稳定注意，别让 LLM 从零推（`Det` 的 inf 即从零推崩）。
- **(7) vulkan 反馈细化**：runner 单独捕获 glslang 报错（区分"shader 运行时编译失败" vs "C++ 编译" vs "forward 崩溃"）；fp16 near-miss 类（误差小而均匀 → 疑似 fp16 路径）。从 §8.1 拆出的进一步细化。

## 附：关键代码位置

| 主题 | 位置 |
|---|---|
| Agent loop / 角色 / 初始信息（自省/范例/机械 param） | `opgen/kernel/kernel_agent.py: run / analyze / _infer_params / _validate_weight_keys` |
| 验证 + arm 非退化门 + PORT BUG 框架 + 多 shape + 输出契约 | `opgen/kernel/kernel_pipeline.py: verify_kernel / arm_forward_overridden / _multishape_check / _size_variants / introspect_model` |
| 失败分类后端感知（E8 覆盖 / arm lane-tail） | `opgen/layer_oracle/failure_taxonomy.py: classify_failure / _coverage / _lane_tail` |
| 三角色 prompt + 领域知识 + 后端背景 | `opgen/kernel/kernel_prompts.py: NCNN_LAYER_BACKGROUND / ARM_LAYER_BACKGROUND / VULKAN_LAYER_BACKGROUND / coder_prompt / debugger_prompt` |
| Profile/Result（含 failure_category、as_backend、shader） | `opgen/kernel/kernel_schemas.py: KernelProfile / KernelResult` |
| 失败分类 taxonomy（纯 numpy）+ 单测 | `opgen/layer_oracle/failure_taxonomy.py: classify_failure`；`opgen/layer_oracle/test_failure_taxonomy.py` |
| 验证 oracle（base/arm）/ vulkan / 接 taxonomy | `opgen/layer_oracle/oracle.py: LayerOracle.verify` / `vulkan_oracle.py: VulkanLayerOracle.verify` |
