# AdapterAgent — 把"算法对的" kernel 改造成"符合 ncnn 契约的" kernel

## TL;DR

AdapterAgent 解决的是一句话诊断出来的问题:**"前面写的算子逻辑对了,但没遵守 ncnn 的规则。"** KernelAgent 写出**数学正确**的 kernel(LayerOracle 绿),但一个 kernel 可以在沙盒里数值自洽、却违反 `ncnn::Net` 的 Layer 契约(weight 的 `mb.load` type 用错、forward overload 与 flag 不符、forward 假设了被 squeeze 的 1D 输入、param-id 跟 pnnx 发出的对不上…),这些只在**端到端(NetOracle)**才暴露。

AdapterAgent 是一个**契约驱动**的修复 agent,只在 e2e 失败时触发,喂给它:
- 一份从 ncnn 源码/官方文档提炼的**权威契约规范**(`ncnn_contract.md`,C1–C6);
- 目标层的**接口字典条目**(param-id / weight load order / flags);
- 该层的**内置 ncnn 实现**作参考(`retrieve_layer_example`);
- 图**实际发给**这个层的 `.ncnn.param` 行(看到真实的 constantA/B/C、transB、size…);
- 当前候选代码 + 具体 e2e 失败。

它被指示**指出违反了哪条契约(C1–C6)或参考实现哪一行,再精确修**——不重设计算法、不猜。输出是与 KernelAgent 同格式的 `{filename: code}`,编排器直接安装。

> **一个诚实的关键结论(见 §4)**:做完 AdapterAgent 后发现,miniset 上 Gemm/BatchNorm 反复挂的**真根因不在 kernel/adapter,而在验证 harness**(缺确定性 seed、LayerOracle 打包器给每个权重写 tag、prompt 示例诱导 mb.load type 用错、production 的 squeeze 不对称、arm 用了下游不跑的 packing)。AdapterAgent 处理**真契约 bug**仍有价值,但这轮真正让 miniset 从 8/9 → 11/11(base 与 arm)靠的是**5 个系统性 harness 修复**。这份文档把两者都记下,因为"以为是 agent 的问题、其实是 oracle 保真度的问题"本身是核心设计教训。

---

## 1. 它做什么(与 KernelAgent / GraphAgent 的分工)

- **KernelAgent**:写算子**实现**(forward 算法),LayerOracle 隔离对拍 PyTorch。见 [KernelAgent.md](./KernelAgent.md)。
- **GraphAgent**:写**图转换**(pnnx pass),让 ncnn 能加载。见 [GraphAgent.md](./GraphAgent.md)。
- **AdapterAgent**:算法已对、但装进 `ncnn::Net` 端到端不对时,把 kernel **改造成符合 ncnn Layer-Net 契约**。它是 KernelAgent 与 NetOracle 之间的"契约翻译"。

在编排器里它接在 **[4b] e2e_repair 循环**——只在 [4] end-to-end numeric 失败时触发,所以已经 e2e 通过的算子完全不碰它(零回归)。见 [OperatorPipeline.md](./OperatorPipeline.md)。

## 2. 为什么需要它:LayerOracle ≠ NetOracle 的语义鸿沟

两个 oracle 验证的**不是同一件事**:
- **LayerOracle**(KernelAgent 用):编译候选 `.cpp` + libncnn.a,`new Cand_<Op>()` 直接跑 forward,对拍 PyTorch。快、轻、不动 ncnn 树。
- **NetOracle**(e2e 用):把层装回 ncnn 源码树、rebuild libncnn、真 `Net::load_param/load_model` + `Extractor` 跑,对拍 PyTorch。这是生产部署行为的 ground truth。

LLM 在 LayerOracle 里满足"自洽"就过,但沙盒对 ncnn 契约的每一处失真,都会让它写出真 Net 里崩的代码。AdapterAgent 的作用就是把这层失真用**显式契约 + 真实 param + 内置参考**补回来,让 LLM 按 ncnn 真规则改。

## 3. 设计:契约驱动的修复引擎

### 3.1 位置与接口(`opgen/orchestrator/adapter_agent.py`)

```
AdapterAgent(task_name, target_layer, class_name, ncnn_root, llm_query, model, run_dir)
  .adapt(code_book, *, ncnn_param_text, e2e_detail,
         input_shapes, expected_out_shape, attempt) -> {filename: code}
```

接在 `operator_agent.py` 的 e2e_repair 循环(受开关 `adapt_e2e`,默认 True 控制):e2e 失败 → 用 AdapterAgent 改造当前 kcode → `install_layer` + `rebuild_libncnn` + 重跑 `_net_numeric`;若 adapter 无改动则退出循环。失败到无法产出代码时,原样返回 code_book(视为"没进展")。

### 3.2 喂给它的四份"真实"(不让 LLM 猜)

1. **ncnn Layer-Net 契约规范** `opgen/ncnn_interface/ncnn_contract.md`——**只**从 ncnn 源码 + `docs/developer-guide/*` 提炼,零臆测。6 条硬契约:
   - **C1 构造 flags**:`one_blob_only` / `support_inplace` 决定用哪个 forward overload(真值表);Gemm 会在 load_param 里按 constantA/B/C 动态翻 flag,要复刻。
   - **C2 load_param**:param-id 是 ncnn 固定编号(≠ONNX/PyTorch 属性序),必须对齐 pnnx 写的;scalar id 0–19,array id = `-23300-idx`。
   - **C3 load_model 的 bin-type 铁律**(最常见 e2e bug):`mb.load(w,0)` 先吃 4 字节 tag(primary/`fwrite_weight_tag_data`);`mb.load(w,1)` 无 tag 直读(secondary/`fwrite_weight_data`,如 bias、BatchNorm 的 slope/mean/var/bias)。哪个权重带 tag 是 layer-specific,查字典 `weights_load_order[i].flag`。
   - **C4 Mat 布局**:w/h/d/c 轴序、`channel(q)` + `cstep`(通道间有对齐 gap,别 flat 索引)、`top.create` 后查 empty、输入 shape 由 pnnx `_ncnn.py` 决定(别硬写单 sample 1D)。
   - **C5 数值/存储 option**:harness fp32 单线程,fp16/bf16/int8 全关。
   - **C6 Net 集成**:retarget 把产出层 type 改成 `Cand_X` 但保留 pnnx 的 param-id;`forward_layer` 按 flag 派发。
   - 附带一段 **Gemm(nn.Linear)worked contract**:constantA=0/constantB=1/constantC=1/transB=1、`B_data=mb.load(K,N,0)`、`Y=alpha·op(A)@op(B)+beta·C`。
2. **目标层接口字典条目**:`render_for_prompt(target_layer, role="kernel")`。
3. **内置 ncnn 实现**:`retrieve_layer_example(ncnn_root, target_layer)`——真 gemm.cpp/innerproduct.cpp 作 load_param/load_model/forward 的参考真相。
4. **真实 `.ncnn.param` 行**:`_e2e_param_text()`——优先读 retarget 后的 param(层实际收到的 param-id/值),让 LLM 看到真实 constantA/B/C、transB、N/K。

### 3.3 输出协议

先要求 3–6 条 bullet **点名违反了哪条 C1–C6 / 参考哪一行、怎么修**(强制归因,反"盲改"),再输出完整文件(fenced,首行裸文件名,与 `extract_kernel_code` 对齐)。保持类名/文件名不变。

## 4. 真正的教训:5 个系统性 harness 根因(比 adapter 更关键)

用 AdapterAgent 跑 Gemm 时它改了两轮仍 e2e 挂(max_diff≈1.9)。停下来做**锚点测试**——直接拿 baseline **真 ncnn Gemm**(不 retarget)跑同管线对比 torch,**也**是 max_diff≈1.6。连官方 ncnn Gemm 都"过不了"→ 证明问题不在我们写的 kernel,而在 harness 对比锚点。顺藤摸出 5 个根因(全部从 ncnn 源码定位,非猜):

| # | 根因 | 现象 | 修复 |
|---|---|---|---|
| 1 | **缺确定性 seed** | `.ncnn.bin` 烤的是 probe 时模型的随机权重,`_net_numeric` 又新建了不同随机权重的模型对比 → 所有带权重算子 e2e 必挂 | 所有 `Model()` 构造前 `torch.manual_seed(0)`(make_pt driver / 各 numeric 对比点)。Gemm max_diff **1.64→0.00056** |
| 2 | **LayerOracle 打包器给每个权重写 tag** | `pack_weights_bin` 每个权重前都写 4 字节 tag,强制 kernel 全 type-0 读;但真 ncnn 只给 primary 写 tag。BatchNorm 全 secondary 用 type-1 读 → 读到 tag 当数据 → var=0 → `1/sqrt(eps)=316` | packer 按字典 per-weight flag 写(`--weight-flag`),bin 与真 .ncnn.bin **byte-identical** |
| 3 | **KernelAgent prompt 示例用 `mb.load(N,1)`** | 诱导 LLM 把 primary weight 也用 type-1 读 → 错位 4 字节 → 99% 错、符号反 | 骨架示例改 `mb.load(N,0)`/primary + `mb.load(M,1)`/secondary;WEIGHTS 段写明 flag→type 规则;E6 加"权重错位"诊断反馈 |
| 4 | **production squeeze 不对称** | production 无条件 drop axis-0(对 Gemm 错:输出是 2D (32,256) 非 batch) | 与 `_net_numeric` 的 pnnx-driven 输入策略对称 drop |
| 5 | **arm 用了下游不跑的 packing** | LayerOracle 对 arm 用 `--packing 4`(elempack=4),但 NetOracle/production 都 `use_packing_layout=false`(elempack=1)→ LLM 被逼写对"打包广播"这条**下游根本不跑**的最难路径 → arm 假阴性 | arm LayerOracle 也用 packing=0,在 arm kernel **实际运行的条件**下验证;prompt 改"packing OFF/elempack=1/support_packing 留 false/4-wide over width" |

> **共同母题**:这 5 个都是 **LayerOracle/harness 的世界 ≠ NetOracle/生产的世界**。每一处失真都让 LLM 在沙盒满足自洽、真环境翻车。修法统一:**让验证条件 = 运行条件**(seed 一致、bin 布局一致、mb.load type 一致、squeeze 一致、packing 一致)。这也是 AdapterAgent 的同源哲学——把真实契约喂进去,而非让 LLM 猜。

结果:**base miniset 11/11**,随后 **base,arm miniset 11/11**(全部 `kernel_arm=success`,0 降级——真 NEON kernel 全过)。

## 5. 设计哲学

- **agent 不是万能补丁,先保证 oracle 保真**。AdapterAgent 只有当 kernel **过了 LayerOracle** 才触发;若 kernel 因 harness 失真在 LayerOracle 就假阴性/假阳性,adapter 根本到不了或修错方向。所以"让 oracle 忠于真 ncnn"优先级高于"加更聪明的 agent"。
- **契约当初始信息给定,不让 LLM 推**。ncnn param-id/weight type/flag 布局这些机械但易错的东西,查字典/读源码算死喂给它,LLM 只负责真正需要判断的改造。(与 KernelAgent §4 同源。)
- **反馈里编码 ncnn 契约,而非泛化 value error**。E6 的"权重错位"提示直接点 mb.load type,让修复循环能**从错误恢复**,而不是在"值不对"里瞎试。
- **失败时回原文档找参考,不盲猜**(用户原则)。adapter prompt 强制归因到 C1–C6/参考行;5 个根因每个都追到 ncnn 源码/文档对应处才修。

## 6. 能力与边界

- **AdapterAgent 覆盖**:真契约 bug——mb.load type/order、forward overload 与 flag 不符、param-id 错、多输入 wiring、输出 shape 契约。
- **AdapterAgent 到不了的**:kernel 在 **LayerOracle 阶段**就挂(如 §4 的 #3 Gemm mb.load type、#5 arm packing)——那要靠 KernelAgent 的 prompt/反馈修,因为 adapter 只在 e2e_repair(kernel 已过 LayerOracle 后)触发。
- **本轮 miniset 上 adapter 实际未触发**:5 个 harness 修复到位后,kernel 直接过 e2e。adapter 作为**真契约 bug 的兜底**保留在管线里,面向后续更复杂/ncnn 无同名层的算子。

## 7. 不需要做的:fp16 验证(已确认)

数据集全 fp32,且生成的 kernel 全部 `support_fp16_storage=false`(默认)。ncnn 契约(`layer-support-behavior.md`)保证 support_fp16_storage=false 的层**绝不会收到 fp16 Mat**——真机即使开 `use_fp16_storage`,ncnn 也在调用前转回 fp32。故我们的 kernel 永远只跑 fp32,当前 fp32 harness 已完整覆盖。fp16 验证**仅当**未来主动生成 fp16 加速 arm kernel(声明 support_fp16_storage=true)时才相关——那是性能优化(OptimizeAgent),非正确性缺口。

## 附:关键代码位置

| 主题 | 位置 |
|---|---|
| AdapterAgent 本体(adapt / prompt 拼装 / 参考实现 / 契约加载) | `opgen/orchestrator/adapter_agent.py` |
| 接入 e2e_repair 循环(adapt_e2e 开关 / `_e2e_param_text` / `_introspect_lite`) | `opgen/orchestrator/operator_agent.py: run / _e2e_param_text / _introspect_lite` |
| ncnn Layer-Net 契约规范(C1–C6 + Gemm worked contract) | `opgen/ncnn_interface/ncnn_contract.md` |
| 接口字典 render / 参考实现 | `opgen/ncnn_interface/lookup.py: render_for_prompt`;`opgen/kernel/kernel_pipeline.py: retrieve_layer_example` |
| 根因1 seed | `graph_pipeline.py`(`_TRACE_DRIVER`/`_BASELINE_NUM_DRIVER`)、`operator_agent._net_numeric_impl`、`kernel_pipeline._build_model`、`production_validation.py`、`validate_layers.py` |
| 根因2 per-weight bin flag | `opgen/layer_oracle/layer_oracle_runner.cpp: pack_weights_bin` + `--weight-flag`;`oracle.py: run/verify(weight_flags)`;`kernel_pipeline.verify_kernel`(从字典算 flags) |
| 根因3 mb.load type prompt + 反馈 | `opgen/kernel/kernel_prompts.py`(骨架示例 / WEIGHTS 段);`failure_taxonomy.py: classify_failure`(weight-misalignment 提示,`has_weights`) |
| 根因4 production 对称 squeeze | `opgen/orchestrator/production_validation.py: production_correctness` |
| 根因5 arm elempack=1 对齐 | `opgen/kernel/kernel_agent.py: self._packing=0`;`kernel_prompts.py: ARM_LAYER_BACKGROUND` |
| 相关记忆 | `memory/project_e2e_seed_rootcause.md`、`memory/project_layeroracle_vs_netoracle_gap.md` |
