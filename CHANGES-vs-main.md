# 相较 main 分支的改动说明(wj 分支)

本分支基于最新的 `main`,在其基础上新增"端到端移动端性能 profiling + 数据集批量转换 + 原生算子经验池种子"等能力。以下按主题记录相对 `main` 的差异。

## 1. 移动端算子 profiling 接入 pipeline

**新增** `ncnn_kernel_test/op_profiler.py` —— 基于 simpleperf 的单算子 PMU profiling 工具(IPC / cache-miss / branch-miss / 算子占比)。

- `profile_operator()` 在 simpleperf record 同一次 benchncnn 运行中**同时解析延迟**(`latency_avg/min/max`)与微架构指标,避免重复测量。
- 新增可选参数 `simpleperf_cmd`(默认 `"simpleperf"`,向后兼容),支持设备自带 / NDK 推送两种 simpleperf 来源。

**修改** `opgen/orchestrator/production_validation.py`：

- 新增自包含方法 `ProductionValidator.profile_op()`：门控(adb 设备 / NDK)→ 交叉编译 benchncnn → 推送 → `_resolve_simpleperf()`（device-first，NDK fallback）→ 委托 `op_profiler` 对 threads=1/2 各采一次，每个 config 同时带微架构指标与延迟。
- **删除** `benchmark()` 方法及死代码 `parse_benchmark_output()`：移动端延迟改由 `profile_op()` 的同一次运行产出，不再单独跑一次 benchncnn。
- 新增 `profile_loop` 字段（simpleperf 下的 benchncnn 循环数，默认 10000）。

**修改** `opgen/orchestrator/operator_agent.py`：

- `_run_production_step` 的 `[5] production` 阶段改为只调 `profile_op()`（不再调 `benchmark()`）。
- 新增 `_perf_from_profile()`：从 profile 的 threads=1 config 提取 `latency_avg/min` 映射成 optimizer 期望的 `{avg, min}`。
- `_run_optimization` 的 `baseline_perf` 改从 `production.profile` 提取（原从 `production.benchmark`）。

## 2. 数据集批量转换（仅存在性检查 + 模型转换，不生成算子）

**新增** `convert_dataset.py`（仓库根）：对 `dataset/Mobilekernelbench` 下每个模型跑 baseline pnnx 转换（复用 `probe_pnnx_ir`），产出 `.ncnn.param/.bin` 到 `dataset/converted/<类别>/<Op>/`，并写汇总 JSON。带 `--keep-work` 开关（默认清理中间产物）。

**新增** `profile_native_baseline.py`（仓库根）：对已转换模型测 **ncnn 原生算子** 真机性能（`retarget_to=None`），通过 `profile_op` 一次运行同时拿延迟 + 微架构指标。

## 3. 原生算子经验池种子（QD 优化器 warm-start）

**新增** `opgen/orchestrator/native_seed.py`：`seed_native_into_pool()` 读 ncnn 原生算子实现，推断 regime + niche cell，作为 floor 种子写入经验池（兵器谱）。

**修改** `opgen/orchestrator/operator_agent.py`：`[6]` 优化阶段前，当算子已被 ncnn 原生支持时调用 `_seed_native()` 注入原生实现作种子。

## 4. 数据集与文档变更

数据集(`dataset/Mobilekernelbench/`):
- **新增** Reduction 类 ArgMax/ArgMin 系列共 8 个模型(`ArgMax.py`、`ArgMin.py` 及其 default_axis / keepdims / negative_axis_keepdims 变体)。
- **删除** `Matrix/Einsum_reduce_dim.py`(47 行)。

清理冗余基线 JSON(改用 `convert_dataset.py` 即时生成,不再随仓库携带):
- **删除** `dataset/Mobilekernelbench_native_support.json`(~2165 行)。
- **删除** `dataset/Mobilekernelbench_pnnx_native.json`(~2972 行)。

文档:
- **新增** `batch_run_all_flow.md`：`batch_run_all.py` 端到端流程说明(各阶段对应的文件与代码位置)。
- **新增** `CHANGES-vs-main.md`(本文件)。

`.gitignore`：新增忽略 `ncnn/`(大型 vendored 源码树,自带 git)与 `dataset/converted/`(可由 `convert_dataset.py` 重新生成)。

## 5. 改动规模(git diff --stat origin/main)

```
 19 files changed, 1504 insertions(+), 5257 deletions(-)
```

净删除行数较多,主要来自两个冗余基线 JSON 的移除;代码净新增集中在 profiling / 转换 / 经验池三块。

## 验证状态

- 所有改动文件 `py_compile` 通过。
- profiling 链路已在真机(adb 连接的 Android 设备)上验证:单次运行同时产出延迟 + IPC 等指标,并正确串入 optimizer 的 `baseline_perf`。
- `convert_dataset.py` 已对 190 个模型实跑:188 转换成功(25 个 ncnn 原生支持)。

