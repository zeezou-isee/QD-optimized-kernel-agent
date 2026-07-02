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

## 2. 移动端 profiling 与延迟的单次测量整合

**修改** `opgen/orchestrator/production_validation.py`：`profile_op()` 在 simpleperf 的同一次 benchncnn 运行中同时产出微架构指标(IPC / cache-miss / branch-miss)与延迟(latency_avg/min/max),不再单独跑一次 benchmark;并吸收了 main 的 decomposed-op retarget 守卫(`expected_src_type`)。

**修改** `opgen/orchestrator/operator_agent.py`：`_perf_from_profile()` 从 profile 的 threads=1 config 提取延迟作 optimizer 基线;`[5] production` 走 profile_op 路径。

> 注:批量转换与原生基线测量脚本(convert_dataset.py / profile_native_baseline.py)未纳入本分支,仅在本地使用。

## 3. 原生算子经验池种子（QD 优化器 warm-start）

**新增** `opgen/orchestrator/native_seed.py`：`seed_native_into_pool()` 读 ncnn 原生算子实现，推断 regime + niche cell，作为 floor 种子写入经验池（兵器谱）。

**修改** `opgen/orchestrator/operator_agent.py`：`[6]` 优化阶段前，当算子已被 ncnn 原生支持时调用 `_seed_native()` 注入原生实现作种子。

## 4. 数据集与文档变更

数据集(`dataset/Mobilekernelbench/`):
- **新增** Reduction 类 ArgMax/ArgMin 系列共 8 个模型(`ArgMax.py`、`ArgMin.py` 及其 default_axis / keepdims / negative_axis_keepdims 变体)。
- **删除** `Matrix/Einsum_reduce_dim.py`(47 行)。

文档:
- **新增** `CHANGES-vs-main.md`(本文件)。

`.gitignore`：新增忽略 `ncnn/`(大型 vendored 源码树,自带 git)与 `dataset/converted/`(转换产物,本地生成)。

## 5. 改动规模(git diff --stat origin/main)

```
 19 files changed, 1129 insertions(+), 97 deletions(-)
```

代码净新增集中在 profiling(op_profiler + production_validation)/ 经验池种子(native_seed)/ orchestrator 融合三块;数据集侧新增 ArgMax/ArgMin、删除 Einsum_reduce_dim。

## 验证状态

- 所有改动文件 `py_compile` 通过。
- profiling 链路已在真机(adb 连接的 Android 设备)上验证:单次运行同时产出延迟 + IPC 等指标,并正确串入 optimizer 的 `baseline_perf`。
- `convert_dataset.py` 已对 190 个模型实跑:188 转换成功(25 个 ncnn 原生支持)。

