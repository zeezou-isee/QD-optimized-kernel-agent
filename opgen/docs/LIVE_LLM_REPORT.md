# graph_agent 在线 LLM 实测报告 + 根因分析

- 日期:2026-06-17
- 模型:`moonshotai/kimi-k2.6`、`z-ai/glm-5.1`(OpenRouter,MoKA 同款调用)
- 用例:`LayerNorm_3d`(nn.LayerNorm,affine=True,对最后两维归一化)
- 结论一句话:**基础设施 + build 反馈链路已打通(build 现在稳定收敛),但 agent 整体仍跑不到成功——卡在 convert 阶段;主因不是"错误文本不够",而是"coder 缺少 PNNX-IR 的事实基准(ground truth)"。**

---

## 一、本阶段排掉的 4 个真实问题(已修)

| 问题 | 现象 | 根因 | 修复 |
|---|---|---|---|
| LLM 调用返回空 / JSON 解析崩 | coder response 0 字节;`JSONDecodeError` | OpenRouter 对慢模型发 `: OPENROUTER PROCESSING` SSE 保活,非流式时解析崩 | `llm_api` 改**流式**累积 content |
| 思考模型空 content | glm/kimi reasoning 烧满预算、`content_len=0`(reasoning 69741 字符) | thinking 模型把 token 全用于推理 | `llm_api` 默认 `reasoning:{enabled:false}`(可 `GRAPH_REASONING=on` 开) |
| 被中断后源码树残留 | 我中途 kill 后 `git status` 脏 | restore 只在 run() 正常结束时跑 | `run()` 加 **try/finally + SIGTERM/SIGINT** 处理,任何退出都回滚 |
| build 反馈太弱、来回震荡 | 修 convert 又把 build 搞崩,循环 | 反馈只是 log 截断,LLM 乱猜头文件 | 接入 `func/Repo_error_location.py`(tree-sitter 定位)+ prompt 写明各 pass 正确 include |

依赖:venv 增装 `tree_sitter` `tree_sitter_cpp` `pyyaml` `ncnn`。

---

## 二、错误定位脚本的效果(已集成)

在 `round_02/build.log` 上实测 `extract_compilation_errors`,输出:
```
[error in another file] file: .../pass_level1/nn_LayerNorm.cpp
  line 25: no member named 'named_input' in 'pnnx::TorchNodeProxy'
  line 26: no member named 'named_input' in 'pnnx::TorchNodeProxy'
--- code ---
  21 |  void write(Operator* op, const TorchGraphProxy& graph, ...) const
  ...
  25 |     op->params["normalized_shape"] = ln->namedInput("normalized_shape");
```
→ 精准给出**文件 / 行号 / 错误信息 / 出错函数代码片段**。已接入 `graph_pipeline.locate_build_errors`(失败回退旧逻辑),用于 build_repair 反馈。另对 convert 的 `map::at` 崩溃加了定向 `annotate_convert_log` 提示。

**集成前后对照(均跑满轮次):**

| 版本 | build 表现 | 是否收敛 |
|---|---|---|
| 集成前(glm3/glm4) | 修 convert 又弄崩 build,**来回震荡** | 否 |
| 集成后(glm6) | **8 轮 build 全部通过、稳定不回退** | build 线已通,但卡 convert |

---

## 三、当前最终状态

`z-ai/glm-5.1`,8 轮,`status=fail`:

| 阶段 | 状态 |
|---|---|
| identify / coder / inject | ✅ 每轮 OK |
| build | ✅ **8/8 轮通过** |
| convert | ❌ 每轮 `std::out_of_range: map::at: key not found`(pass_level2) |
| structural / numeric | 未到达 |

源码树每轮**自动回滚干净**。

glm 生成的 `pass_level2/F_layer_norm.cpp` 的 PNNX-IR pattern 写错:`aten::layer_norm` 实际有 **6 个输入**(input, normalized_shape, weight, bias, eps, **cudnn_enabled**),glm 只写了 5 个、operand 计数也不符 → 匹配后 `captured_params.at("normalized_shape")` 取不到键而崩溃。即便喂了定向 hint,glm-5.1 连续 8 轮都没写对这个 pattern。

---

## 四、根因分析:agent 为什么还不行?是反馈不够吗?

**部分是,但不是全部。更准确地说:反馈在"编译错误"这条线已经足够;真正的瓶颈是 coder 写 PNNX-IR pattern 时缺少"事实基准",这是一个 grounding(上下文)问题,不只是"错误文本多寡"。**

分四层拆解:

### 1)反馈的"可定位性"分两类,我们只解决了一类
- **编译错误**:有 `文件:行:列`,tree-sitter 能精确定位 → 已用脚本解决,build 因此收敛。
- **convert 运行时崩溃**:是 C++ 异常(`map::at`),**没有文件/行**,无法定位到是哪一行 `.at()`、哪个 key。我的 hint 只能给"这一类错误的通用修法",指不到具体位置 → coder 仍要靠猜。
- 👉 所以"加错误文本"对 convert 帮助有限——**问题不在错误描述不够细,而在错误本身不可定位**。

### 2)最关键:coder 从未看到"要匹配的真实 IR"(grounding 缺失)
agent 让 LLM **凭空**写 `match_pattern_graph`(要精确到 aten 算子的输入个数/顺序/常量),却从没给它**该算子在 pnnx 里的真实样子**。
- pnnx 本身会产出 `.pnnx.param`(中间 IR),里面**明确写着** `aten::layer_norm` / `nn.LayerNorm` 的确切 operands 和 params——这正是 pattern 要对齐的"标准答案"。
- 当前 agent 没有把这个真实 IR 喂回给 coder,导致它反复猜错输入个数(漏 `cudnn_enabled`)。
- 👉 **这才是核心缺口:不是"反馈不够",而是缺少"输入侧的事实基准"。** 正确做法是在生成前先跑一遍 pnnx 拿到目标算子的真实 pnnx IR(或 aten 节点签名),作为 coder 的 grounding。

### 3)检索到的范例不是"最近邻"
`retrieve_examples` 给 LayerNorm 拉的是 `F_batch_norm`/`F_group_norm`,**没拉到最该参考的 `F_layer_norm`**(因为 analyzer 的 `analog_ops` 没指它)。对一个 ncnn 已支持的算子,正确范例就在仓库里却没被喂进去 → coder 等于"没有标准答案还要重写标准答案"。

### 4)模型能力 + DSL 冷门
PNNX-IR 是很冷门的内部 DSL,`glm-5.1`(且关了 thinking)对它的先验弱。即使反馈到位,它在"算子签名级精确度"上也容易出错。更强模型(Claude/GPT)或开 thinking 大概率更好,但不该让 agent 的成败押在模型记忆上——应靠 grounding 把"该匹配什么"变成已知事实。

### 小结
| 维度 | 现状 | 是否瓶颈 |
|---|---|---|
| 循环/注入/构建/回滚 基础设施 | 已稳 | 否 |
| 编译错误反馈 | 已精确(脚本) | 否(已解决) |
| convert 错误可定位性 | 运行时崩溃,无法定位 | 是(但靠"加文本"治不好) |
| **coder 的 IR grounding** | **缺失** | **是(核心)** |
| 最近邻范例检索 | 不准 | 是(次要) |
| 模型对 PNNX-IR 的能力 | 偏弱 | 是(可换模型缓解) |

---

## 五、据此的改进方向(供下一步选择)

1. **给 coder 喂真实 pnnx IR(最高优先)**:生成前先 `pnnx model.pt`(不加新 pass)拿到 `.pnnx.param`,把目标算子的真实节点(operands/params/attrs)注入 coder/debugger prompt,让 pattern 有据可依。
2. **最近邻范例**:analyzer 输出更准的 `analog_ops`,或用算子名直接在 `pass_level2/pass_ncnn` 里 grep 同名/同类已有 pass 喂给 coder。
3. **convert 崩溃定位增强**:崩溃时自动用 pnnx 的调试输出 / 缩小输入,或在 write() 失败前先校验"pattern 捕获的 key 集合 ⊇ write() 用到的 key 集合"(可在注入前做静态检查,把 `.at()` 的 key 与 pattern 的 `%capture` 比对,提前报错而非运行时崩)。
4. **更强 coder 模型**或开 thinking(`GRAPH_REASONING=on`)。
5. **真正未支持的算子**做最终验证(LayerNorm 已支持,适合验机制;验"凭空写新算子"应换一个 ncnn 没有的算子)。

---

## 六、第二轮设计改进(根据根因)+ 关键更正

根据"coder 缺 grounding"的根因,对 agent 做了以下改动并重跑:

| 改动 | 文件 | 作用 |
|---|---|---|
| **pnnx IR 探针 grounding** | `graph_pipeline.probe_pnnx_ir` | 生成前先跑基线 pnnx,拿到真实 `.pnnx.param`(算子真实类型/参数/attrs)注入 coder/debugger prompt |
| **最近邻范例** | `retrieve_examples(op_types=...)` | 按 IR 真实算子类型 grep 现有 pass 作范例 |
| **prompt grounding 段 + 最小改动指引** | `graph_prompts` | 告诉 coder 真实 IR、"无 aten 残留就别写 pass_level1/2" |
| **teardown 重编 pnnx** | `graph_agent` | restore 只还原源码,二进制会残留坏 pass;补一次重编保证二进制也干净 |
| **make_pt/run_conversion 绝对路径** | `graph_pipeline` | 修相对路径在 cwd 下被重复拼接的 bug |
| **baseline 支持检测 + 短路** | `probe_pnnx_ir` + `graph_agent` | 基线若已结构+数值正确,直接报 `already_supported`,不造 pass |

### ⚠️ 关键更正:LayerNorm_3d 其实**已被正确支持**
- 用**同一份输入**测基线 pnnx→ncnn:`max diff 0.00198, allclose True`。
- 之前我说"baseline 数值错(diff 5.79)"是**测试脚本输入不一致的 bug**(ref 与 ncnn 用了不同随机输入),特此更正。
- 即:`LayerNorm_3d`(nn.LayerNorm 对最后两维 + affine)pnnx 映射成 `ncnn LayerNorm 0=1024 1=1e-5 2=1` 是**数值正确**的。

### 这解释了 glm 为何一直"失败"
agent 在给一个**本就能正确转换**的算子重写 pass,glm 反而把能用的 baseline 改崩(多写会崩的 pass_level2)。**不是 agent 框架的问题,是测试用例选错了 + 模型能力。**

### 改进后重跑结果(glm-5.1)
```
[agent] pnnx IR probe: op_types=['nn.LayerNorm'] residual_aten=[] baseline_supported=True
[agent] operator already supported by current pnnx/ncnn — no new pass needed (verify-only).
status = already_supported
```
→ agent 现在能**正确识别"已支持"并干净退出**(~1 分钟,不调 LLM、不破坏源码树)。

### 结论
- agent 框架 + 反馈链路(grounding / error-locator / 最近邻 / 干净回滚)已显著增强且工作正常。
- **要真正验证"凭空写新算子",必须换一个 `baseline_supported=False` 的算子**(pnnx/ncnn 确实转不出或转错的)。LayerNorm_3d 不满足。
- 下一步建议:用 probe 批量扫数据集,挑出 `baseline_supported=False` 的算子作为真实 authoring 测试用例。
