# `batch_run_all.py` 执行流程

本文档记录运行 `batch_run_all.py` 时经历的具体步骤,以及每一步对应的**文件**与**位置**。

## 总体两层结构

`batch_run_all.py` 本身是**外层批量调度器**,真正的算子处理流程在它为每个算子启动的**子进程**里(`opgen/cli/run_operator_agent.py` → `OperatorAgent.run()`)。

```
batch_run_all.py                         外层批量调度(发现算子 / 断点续跑 / 起子进程 / 收结果)
  └─ subprocess: opgen/cli/run_operator_agent.py   CLI 入口(只解析参数)
       └─ opgen/orchestrator/operator_agent.py     编排核心 run():按 [1]~[7] 阶段串联
            ├─ [1]/[1b] KernelAgent     opgen/kernel/kernel_agent.py
            ├─ [2]      NetOracle 装层+重建  opgen/layer_oracle/net_oracle.py
            ├─ [3b]     GraphAgent       opgen/graph/graph_agent.py
            ├─ [4]      NetOracle 端到端数值
            ├─ [5]      ProductionValidator  production_validation.py
            ├─ [6]      OptimizeAgent(可选)
            └─ [7]      Cleanup / Register
```

---

## 第 0 层:批量调度 — `batch_run_all.py`

| 步骤 | 做什么 | 位置 |
|---|---|---|
| 发现算子 | 扫描 `dataset/Mobilekernelbench_subset/**/*.py`,得到 `(category, op)` 列表 | `batch_run_all.py:44` `discover_ops()` |
| 断点续跑 | 读 `batch_all_results_arm.json`,已有终态(非 crash/timeout)的算子跳过 | `batch_run_all.py:54` `load_results()` + `:138` 跳过判断 |
| 逐个起子进程 | 对每个算子 `subprocess.run([python, opgen/cli/run_operator_agent.py, --task <op>, ...])`,30 分钟硬超时(`PER_OP_TIMEOUT=1800`) | `batch_run_all.py:89` `run_one()` |
| 收集结果 | 读该算子写出的 `opgen/runs/<op>/operator/summary.json`,抽取各阶段状态写回 json | `batch_run_all.py:67` `summarize()` |
| 汇总 | 统计 success 总数并打印 | `batch_run_all.py:151` |

子进程命令固定参数(`batch_run_all.py:89-99`):

- `--max-rounds 15`(kernel 最大轮数)
- `--graph-max-rounds 10`(graph 最大轮数)
- `--backends base,arm`(同时生成 base + ARM NEON kernel)
- `--compile-mode build_lib`(复用 `libncnn.a`)

LLM 后端通过环境变量配置(`batch_run_all.py:30-37`):`LLM_BACKEND=deepseek` 走 DeepSeek 直连(需 `DEEPSEEK_API_KEY`),默认走 OpenRouter(需 `OPENROUTER_API_KEY`)。

> 注意:batch 默认**不**传 `--optimize` 也**不**传 `--install`,所以 [6] 优化阶段会跳过,[7] 走 **CLEANUP** 分支(临时验证后还原源码树)。`--end-to-end` 默认开启,故 [2][4][5] 都会执行。

---

## 第 1 层:CLI 入口 — `opgen/cli/run_operator_agent.py`

只做 `argparse` 参数解析,然后构造并运行 `OperatorAgent(...).run()`。

- 参数解析:`run_operator_agent.py:24-72`
- 构造 + 运行:`run_operator_agent.py:74`(`OperatorAgent(...).run()`)

所有真正的编排逻辑都在 `run()` 里。

---

## 第 2 层:编排核心 — `opgen/orchestrator/operator_agent.py` 的 `run()`(`:117`)

按顺序执行各阶段(代码里用 `===== [n] ... =====` 标注):

| 阶段 | 做什么 | 位置 | 委托给谁 |
|---|---|---|---|
| **[1] Kernel** | KernelAgent 从零写 base 版 ncnn layer kernel,对 PyTorch 数值验证;失败则中止 | `operator_agent.py:122-139` | `KernelAgent(...).run()` → `opgen/kernel/kernel_agent.py` |
| **[1b] ARM kernel** | (因 `--backends base,arm`)写 NEON/NC4HW4 子类 kernel;默认 arm 是硬门槛,失败即整体失败(除非 `--allow-backend-fallback`) | `operator_agent.py:144-172` | `KernelAgent(..., backend="arm")` |
| **[2] Bridge** | 把 kernel 装进 `ncnn/src/layer/`,重新编译 `libncnn.a`;失败则还原+重建后中止 | `operator_agent.py:178-195` | `NetOracle.install_layer()` / `rebuild_libncnn()` → `opgen/layer_oracle/net_oracle.py:179` / `:222` |
| **[3] 存在性检查** | 跑 baseline pnnx,若 ncnn 已原生支持该算子则跳过 GraphAgent | `operator_agent.py:206-221` + `_check_already_in_ncnn()` `:314` | pnnx 探针 |
| **[3b] GraphAgent** | 从零写 pnnx 计算图转换 pass,强制目标=新层 `cls`;`graph-max-rounds`(=10)内不收敛则中止 | `operator_agent.py:224-240` | `GraphAgent(..., force_target_layer=cls).run()` → `opgen/graph/graph_agent.py` |
| **[4] 端到端数值** | 用转换后的 `.ncnn.param/.bin` 跑整模型,对 PyTorch allclose | `operator_agent.py:243-248` + `_net_numeric()` `:489` | `NetOracle` |
| **[5] Production 验证** | MoKA 式生产编译 + 正确性(+可选 benchmark);`build_lib` 复用 libncnn | `operator_agent.py:251-252` + `_production_validation()` `:371` / `_run_production_step()` `:384` | `ProductionValidator` → `production_validation.py` |
| **[6] Optimization** | 仅当 `--optimize` 开启且功能全过,才驱动 OptimizeAgent(MAP-Elites / 线性)优化性能 | `operator_agent.py:262-273` + `_run_optimization()` `:414` | OptimizeAgent |
| **[7] Cleanup / Register** | `finally` 块:默认(无 `--install`)还原 ncnn 源码树并重建干净 libncnn;若 `--install` 且全过则永久注册算子到 ncnn/pnnx | `operator_agent.py:274-292` | `NetOracle.restore()` `:213` |
| 写 summary | 把整个 `summary` 写到 `runs/<op>/operator/summary.json`,供 batch 层读取 | `operator_agent.py:308` | — |

### 失败即中止的关键门槛

- [1] kernel 失败 → 整体 `fail`(`:136`)
- [1b] arm 失败且无 fallback → 整体 `fail`(`:160-172`)
- [2] libncnn 重建失败 → 还原后 `fail`(`:189-195`)
- [3b] graph 不收敛 → `fail`(`:236-240`)

---

## Kernel 生成之后:编译 → 模型转换 → 正确性检验 → 性能测试

KernelAgent 生成算子后,这四步**都会做**,分别对应 `run()` 的 `[2]` / `[3b]` / `[4]+[5]` / `[5]` 阶段。每个阶段委托给具体实现类。下表行号基于当前源码(存在性检查已在 `[3]`、bridge 之后)。

| 步骤 | 阶段 | 入口(operator_agent.py) | 实现 |
|---|---|---|---|
| 编译 | [2] | `:182`(装层)`:185`(重编) | `net_oracle.py:179` `install_layer` / `:222` `rebuild_libncnn` |
| 模型转换 | [3b] | `:227` | `graph_agent.py` / `graph_pipeline.py:615` `run_conversion` |
| 正确性(端到端) | [4] | `:246` | `operator_agent.py:542` `_net_numeric_impl` + `net_oracle.py:238` `run_net` |
| 正确性(生产) | [5] | `:254` | `production_validation.py:139` `production_correctness` |
| 性能测试 | [5] | `:426`(`--benchmark` 开关) | `production_validation.py:191` `benchmark` |

### 1. 编译(把 kernel 装进 ncnn 并重编库)

阶段 **[2] BRIDGE**,`operator_agent.py:176-197`:

- 装层:`netoc.install_layer(kcode, cls)` → `operator_agent.py:182`(arm 子类 `:184`);实现 `net_oracle.py:179`(拷 `.cpp/.h` 到 `src/layer/` + 插 `ncnn_add_layer`)
- 重编:`netoc.rebuild_libncnn()` → `operator_agent.py:185`;实现 `net_oracle.py:222`(`cmake --build` 重编 `libncnn.a`)

> 第二道"生产编译"门在 [5]:`ProductionValidator.production_compile()`(`production_validation.py:103`)。`build_lib` 模式只校验 `libncnn.a` 存在(`:105-111`);`build_full` 模式做 MoKA 式完整 ncnn 构建(`:113-133`)。

### 2. 模型转换(PyTorch → ncnn,pnnx pass)

阶段 **[3b] GraphAgent**,`operator_agent.py:224-242`:

- 调用:`GraphAgent(..., force_target_layer=cls).run()` → `operator_agent.py:227-231`
- 实现:`opgen/graph/graph_agent.py`,底层转换 `opgen/graph/graph_pipeline.py:615` `run_conversion`(产出 `.ncnn.param/.ncnn.bin`)
- 若 ncnn 已原生支持该算子([3] 存在性检查命中 `:209`),**跳过转换**、复用 baseline pnnx 转换产物(`operator_agent.py:218-223`,探针 `_check_already_in_ncnn()` `:325`)

### 3. 正确性检验(两层,均对 PyTorch allclose)

**(a) [4] 端到端数值**(整网功能),`operator_agent.py:244-250`:

- 调用:`self._net_numeric(netoc, graph_sum, op_class=cls)` → `:246`
- 实现:`_net_numeric()` `operator_agent.py:531` → `_net_numeric_impl()` `:542`;用 `.ncnn.param/.bin` 跑整网(`netoc.run_net()` `net_oracle.py:238`),`np.allclose(atol=2e-3)`(`:587`)
- 关键:`retarget_param_output_file(param, rp, op_class)`(`:556`)把输出层重定向到自研 `Cand_<Op>`,确保测的是自己的算子(对已存在算子尤其重要)

**(b) [5] 生产正确性**,`operator_agent.py:252-254`:

- 调用:`self._production_validation(...)` → `:254`,内部 `_run_production_step()` `:426`
- 实现:`ProductionValidator.production_correctness()` `production_validation.py:139`(同样 NetOracle 跑整网 + 重定向 + allclose,`:172/:181`)

### 4. 性能测试(benchmark,默认关闭)

阶段 **[5]** 内,仅当 `--benchmark` 且正确性通过时执行:

- 触发:`_run_production_step()` `operator_agent.py:426`,`self.do_benchmark and _mandatory_ok` 时调用 `pv.benchmark(...)`
- 实现:`ProductionValidator.benchmark()` `production_validation.py:191`(MoKA 式:交叉编 benchncnn → adb push → 真机跑整网 `.ncnn`)
- 无 `ANDROID_NDK` 或无设备则优雅 SKIP(`:201-207`),**不算失败**;同样用 `retarget_param_output_layer`(`:215`)保证测自研算子
- 另有迭代用的性能路径:`[6]` 优化阶段(opt-in)的 `optimize/evaluator/measure_harness.py`,宿主机隔离 LayerOracle runner 测延迟,不经 benchncnn

> **batch 默认行为**:`batch_run_all.py` 不传 `--benchmark` 也不传 `--optimize`,所以实际跑的是 **编译 → 转换 → 正确性(端到端 + 生产编译/正确性)**;性能 benchmark 与 [6] 优化阶段默认跳过。

---

## 关键文件速查

- 批量调度:`batch_run_all.py`
- CLI 入口:`opgen/cli/run_operator_agent.py`
- 编排核心:`opgen/orchestrator/operator_agent.py`(`run()` 在 `:117`)
- Kernel 生成:`opgen/kernel/kernel_agent.py`
- 图转换:`opgen/graph/graph_agent.py`
- 装层 / 重建 / 还原:`opgen/layer_oracle/net_oracle.py`
- 生产验证:`opgen/.../production_validation.py`

## 产物目录

```
opgen/runs/<task>/
  kernel/        KernelAgent 产物(analyzer/profile/round_XX/*.h/.cpp/result.json)
  graph/         GraphAgent 产物(pnnx_ir_probe/round_XX/*.ncnn.param ...)
  operator/      总控合并 summary.json   ← batch 层读取此文件
```
