# graph_agent 设计方案（待审）

## Context（为什么做这个）

旧 MoKA Agent 只能为 ncnn **已有**算子改 kernel，无法引入框架里**不存在**的新算子——因为新算子必须先有「PyTorch → ncnn 计算图转换」(PNNX pass) 才能生成 `.ncnn.param/.bin`，否则 MoKA 的编译/正确性/性能流程第一步就断。

`graph_agent` 就是补这一能力：一个独立可调用的 `GraphAgent` 类，自动完成 **算子类型识别 → 生成转换 pass → 注入 ncnn 源码树 → 构建 pnnx → 跑转换 → 验证正确性 → 出错反馈并迭代修复** 的完整闭环。

### 已确认的前提（来自用户）
1. **pnnx 从源码构建**（环境已有 libtorch）——agent 负责 cmake+make，注入新 pass 后增量重编。
2. **kernel 由别的 agent 负责**——graph_agent 只 own 转换层；验证做两级（结构验证永远做；数值 allclose 在 kernel 可运行时才做）。
3. **完全独立自成一套**，但可复用 `EndtoEndMobilekernelAgent/tools/`（read_file/write_file/edit_file/glob_search/grep_search/bash_exec）。所有产物写在 `EndtoEndMobilekernelAgent/` 下。

### 关键事实（来自源码核实）
- `ncnn` 与 kernel 解耦的两条验证路径：layer 级 C++ 单测（`tests/test_*.cpp` 的 `test_layer()`，不需转换）验证 kernel；端到端 `pnnx/tests/ncnn/test_*.py`（PyTorch→ncnn→allclose）验证转换。
- **pnnx 的 pass 源文件在 `tools/pnnx/src/CMakeLists.txt` 里显式列举**（非 glob，64 个 `F_/nn_/Tensor_`）→ 注入新 pass **必须同时 patch CMakeLists.txt**。
- 测试在 `tools/pnnx/tests/ncnn/CMakeLists.txt` 用 `pnnx_ncnn_add_test(name)` 注册。
- pnnx 当前未编译。

---

## 目录结构（全部新建于 EndtoEndMobilekernelAgent/）

```
EndtoEndMobilekernelAgent/
├── graph_agent.py        # GraphAgent 类：agent loop（状态机）
├── graph_pipeline.py     # 功能函数：identify/inject/build/convert/verify/restore（可独立调用）
├── graph_schemas.py      # GraphTask / GraphResult / OpProfile 数据类
├── graph_prompts.py      # 角色 prompt：analyzer / coder / debugger(三模式)
├── llm_api.py            # 轻量 LLM 封装（openrouter，独立实现）
├── config.py             # 路径配置：ncnn_root / pnnx_dir / libtorch / 数据集根
├── run_graph_agent.py    # CLI 入口 + 用例
├── tools/                # 已存在，复用
└── runs/<task>/          # 每个任务的产物（prompt/response/plan/各阶段日志）
```

---

## 一、功能函数层 GraphConversionPipeline（graph_pipeline.py）

把转换流程拆成离散、可单测的函数（这是「agent loop + 功能函数调用」的功能函数侧）：

| 函数 | 作用 | 失败反馈 |
|---|---|---|
| `identify_op(model_code)` → `OpProfile` | 识别算子来源形态与目标 ncnn layer，决定要写哪些 pass | — |
| `retrieve_examples(profile)` → `dict` | 用 grep/glob 从 pass_ncnn/level2 找最相似的已有 pass + param/weight 表，喂给 coder | — |
| `inject_files(code_book)` → `(ok, backup)` | 备份→写新 pass 文件→patch 两个 CMakeLists | 哪个文件/哪段 CMake patch 失败 |
| `build_pnnx()` → `(ok, log)` | cmake+make（增量） | `extract_build_errors()` 抽错误+上下文，过滤到新 pass 文件 |
| `run_conversion(pt, inputshape)` → `(ok, paths, stdout)` | 跑 pnnx 二进制，产出 `.pnnx.param/.ncnn.param/.bin` | pnnx stdout 的 unsupported/no-rewrite 告警 |
| `verify_structural(pnnx_param, ncnn_param, profile)` → `(ok, report)` | 解析中间/最终 param | 残留 `aten::/prim::`（=level2 没匹配）/ 目标 layer 缺失（=pass_ncnn 没匹配） |
| `verify_numeric(test_py)` → `(ok, report)` | 跑生成的 test：PyTorch vs ncnn `allclose` | 每个输出的 max abs diff / shape 不符 → 参数或权重映射错 |
| `restore_files(backup)` | 删新建文件 + 还原 CMakeLists，保证源码树干净 | — |

> 注意与 MoKA `restore_files` 的差异：MoKA 只覆盖已有文件；这里要**新建文件**，所以 restore = 删除新增文件 + 回滚被 patch 的 CMakeLists（备份原文件 + 记录新增清单）。

### OpProfile（算子类型识别产物）
依据 `ncnn_graph_conversion.md` 的分类，从 PyTorch 代码识别出：
- `source_form`: `nn_module` / `functional` / `aten` / `composite`（决定要不要 pass_level1、pass_level2）
- `category`: `unary` / `binary` / `weighted`(conv/linear/bn) / `tensor_manip` / `composite`
- `target_ncnn_layer`: 目标 ncnn layer 类型名（用于结构验证 & write() 的 type_str）
- `needs_weight`: 是否带权重（决定用 `write(op,params)` 还是 `write(op,params,attrs)`）
- `rank_coverage`: 测试要覆盖的输入维度（1D~4D）
- `files_to_write`: 推断出的待写文件清单
- `analog_ops`: 最相似的已有算子（供模仿）

---

## 二、Agent loop（graph_agent.py，GraphAgent 类）

状态机：**pipeline 结果驱动**，与 MoKA 同构但阶段不同。

```
round 0  (phase = identify_and_generate)
    profile  = identify_op(model_code)              ← analyzer 角色(LLM)
    examples = retrieve_examples(profile)           ← 工具检索(grep/glob)
    code     = coder(profile, examples)             ← coder 角色(LLM) 产出 pass + test
    result   = run_conversion_pipeline(code)        ← inject→build→convert→verify
    save_round + update_memory

round 1..max_rounds-1
    phase = choose_next_phase(result):
        not inject_ok      → "inject_repair"        (一般是 CMake/文件名问题，多为可自动修)
        not build_ok       → "build_repair"         ← debugger:编译错误
        not convert_ok     → "convert_repair"       ← debugger:残留 aten / 没匹配到
        not structural_ok  → "convert_repair"       ← debugger:param 不对
        not numeric_ok     → "numeric_repair"       ← debugger:allclose 失败
        else               → success(break)

    plan = debugger(phase, result)                  ← 针对性反馈(见下) → 修复计划
    code = coder(code_book=result.code, plan=plan)  ← coder 据计划重写
    result = run_conversion_pipeline(code)
    save_round + update_memory
    if all ok: break

收尾：keep_changes_on_success 配置决定保留还是 restore；写 summary.json
```

### run_conversion_pipeline(code)（一轮内的固定编排）
```
inject_files(code) → build_pnnx() → run_conversion() → verify_structural() →（kernel 可运行?）verify_numeric()
任一步失败即短路返回，并 restore_files() 保持源码树干净（成功时按配置保留）
```

### 三个角色（graph_prompts.py）
同一 LLM 在不同 prompt 下扮演：
- **analyzer**：输入 PyTorch 代码 → 输出 OpProfile（结构化 JSON）。
- **coder**：输入 OpProfile + 检索到的相似 pass 示例 + param/weight 表 → 输出 pass cpp(可多变体) + test py。
- **debugger（三模式）**：build_repair / convert_repair / numeric_repair，各自拿到对应的针对性反馈 + 当前代码 + memory → 输出修复计划。

---

## 三、反馈信息设计（出错怎么让 agent 优化）—— 这是质量关键

利用「同时拥有中间产物 `.pnnx.param` 和最终 `.ncnn.param`」精确定位是哪一段 pass 坏了：

| 失败阶段 | 反馈内容 | 指向 |
|---|---|---|
| inject | 哪个文件写失败 / 哪个 CMakeLists 段没插入成功 | 注入逻辑（多可自动修，不必走 LLM） |
| build | `extract_build_errors`：抓 `error:`/`undefined reference`/`ld:` + 上下文，过滤到新 pass 文件 | C++ 语法 / API 用错 |
| convert | `.pnnx.param` 里残留 `aten::xxx/prim::` 节点 | **pass_level2 没覆盖该导出形态** → 补匹配变体 |
| convert | `.ncnn.param` 里没有目标 layer 类型 | **pass_ncnn 没匹配/type_str 错** |
| convert | pnnx stdout 的 `unsupported`/`no rewrite` 告警 | 提示具体哪个 op 没转 |
| numeric | 每个输出的 max/mean abs diff、shape mismatch、哪个输出 index 错 | **write() 里 param 编号/权重布局填错** |

---

## 四、文件注入的具体动作（firm）

新算子 `MyOp` 一轮注入：
1. 写 `tools/pnnx/src/pass_ncnn/F_my_op.cpp`（必需）、按需 `pass_level2/F_my_op.cpp`、`pass_level1/nn_MyOp.cpp`。
2. **patch `tools/pnnx/src/CMakeLists.txt`**：把新文件加进对应的 `set(pnnx_pass_ncnn_SRCS ...)` / `pnnx_pass_level2_SRCS` / `pnnx_pass_level1_SRCS` 列表。
3. 写 `tools/pnnx/tests/ncnn/test_F_my_op.py`（端到端测试，覆盖多 rank/多等价写法）。
4. **patch `tools/pnnx/tests/ncnn/CMakeLists.txt`**：加 `pnnx_ncnn_add_test(F_my_op)`。
5. 备份被改的两个 CMakeLists + 记录新增文件清单 → 供 restore。

构建：`cd tools/pnnx && cmake -B build (-DTorch_DIR=...) && cmake --build build -j`（首次慢，之后增量）。
转换：从用户 Model + `get_inputs()` 生成 trace 脚本得 `.pt`，跑 `build/src/pnnx model.pt inputshape=[...]`。

---

## 五、验证（end-to-end 怎么测）

- **结构验证（永远做，不依赖 kernel）**：解析 `.pnnx.param` 确认目标 op 无残留 torch 域节点；解析 `.ncnn.param`（文本：magic / layer_count blob_count / 每行 `type name nin nout ...`）确认目标 layer 出现且 param 合理。
- **数值验证（kernel 可运行时做）**：跑生成的 `test_F_my_op.py`，`torch.allclose(rtol=1e-4, atol=1e-4)` 比对 PyTorch 与 ncnn 输出；或直接 `ctest -R test_ncnn_F_my_op -V`。
- kernel 不存在/跑不起来 → numeric 标记 skipped，以结构验证为准，并在 GraphResult 里标注。

GraphResult 字段：`task_name, op_profile, identify_ok, inject_ok, build_ok, convert_ok, structural_ok, numeric_ok, generated_code, build_error, convert_log, verify_log, artifacts, messages`。

---

## 六、独立可调用性（满足"方便单独调用验证"）

- `GraphAgent(task_name, model_code/py_path, ...).run()` 一键跑完整闭环。
- `GraphConversionPipeline` 的每个函数可单独 import 调用（如只想测 `verify_structural` 或 `build_pnnx`）。
- `run_graph_agent.py` 提供 CLI：`python run_graph_agent.py --task MyOp --model path/to/MyOp.py --max-rounds 8 [--keep-on-success] [--no-numeric]`。

---

## 七、实现顺序（建议）

1. `graph_schemas.py` + `config.py`（数据契约与路径）
2. `graph_pipeline.py` 的 inject/build/convert/restore（纯机械，先打通源码树注入与编译）
3. `verify_structural` + `verify_numeric`
4. `llm_api.py` + `graph_prompts.py`（analyzer/coder/debugger）
5. `graph_agent.py` loop 串起来
6. `run_graph_agent.py` CLI + 用 HardSigmoid 这类已支持算子做"回归测试"（应一轮通过），再拿一个未支持算子做真实验证。

---

## 已确认的范围决定（首版）
- **算子范围**：首版只做**无权重算子（unary / functional）**，如 hardsigmoid/elu/激活函数类；weighted(conv/linear/bn)、tensor_manip、composite 二期。
- **验证级别**：**结构验证 + 数值 allclose 都做**（数值验证需 kernel 可运行，由别的 agent 提供 layer）。
- **max_rounds 默认 8**。
- 决定：按本设计开始落代码，实现顺序见第七节。
