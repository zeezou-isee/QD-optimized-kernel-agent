# Miniset 真机测试 — 最终结果(P0–P3)

设备:Redmi(骁龙 778G)· CPU Cortex-A78/A55 · GPU Adreno 642L · 全部性能真机测得(从不 Mac 测最终 perf)。

---

## P0 — device-in-the-loop 测量 ✅ + 证据实验 ✅(部分)

**测量后端**:`scripts/bench_miniset_device.py`(CPU:install→交叉编译 benchncnn→retarget→真机 profile→还原树)+ `scripts/bench_vulkan_device.py`(GPU:交叉编译 vulkan oracle runner→推真机→Adreno 上运行时编译 shader→跑)。`op_profiler` 一次 simpleperf run 同出 PMU+延迟;vulkan runner 新增 `--bench N` 出 GPU 延迟。

**关键方法学发现**:benchncnn 默认 `use_fp16 + packing = true` → ncnn 原生跑 fp16+NC4HW4 优化路径,我们的 baseline 是 fp32 标量 → 不同档位不可直接对比。公平对比需同档位(fp32)+ 放大 shape 让 kernel 主导。

**公平 fp32 对比(真机 CPU,min ms)**:
| op | native fp32 | ours(fp32) | ratio |
|---|---|---|---|
| Abs | 4.63 | 2.26 | **0.49×(我们快 2×)** |
| BatchNorm | 4.63 | 2.33 | **0.50×** |
| ReduceSum | 3.59 | 0.44 | **0.13×(快 8×)** |
| AveragePool | 0.21 | 0.24 | ~1.0× |

→ **同档位(fp32)下,我们从零生成的 baseline 已追平/反超 ncnn 的 fp32 回退路径 2–8×**。ncnn 的真正优势在 fp16+packing(见 P2)。

**证据实验 — QD vs best-first(Conv,base)**:verdict = **"qd"** — QD 15.20ms vs best-first 15.57ms,**argmin 来自非主流 niche(多样性付费成立)**。直接支撑"为什么用 QD"。

**证据实验 — wiki on/off**:🚫 被一个**既有 bug 阻塞**:optimize 的 arm evaluator 跑不了带权重/functional 算子的 baseline(`load_model failed` / `forward ret=-100`)。wiki 只对 arm/vulkan 后端生效,故此消融待该 bug 修复。

## P1 — vulkan 真机进流程 ✅(8/10)

从零生成的 vulkan kernel **在真 Adreno GPU 上运行时编译 shader 并执行**,延迟 + 正确性(vs host MoltenVK 参考,已验证 == torch):
| op | GPU min ms | 正确性(max_diff) |
|---|---|---|
| Abs | 0.19 | 0.0 |
| Add | 0.19 | 0.0 |
| And | 0.08 | 0.0 |
| AveragePool | 0.22 | 0.0 |
| BatchNorm | 0.53 | 0.0 |
| Conv | 0.54 | 3.8e-6 |
| Mul | 0.30 | 0.0 |
| ReduceSum | 0.24 | 0.0 |
| Gemm | ✗ | shader 编译失败(已知坏 shader) |
| Greater | ✗ | 用了 multi-shader API,harness 未设 CANDIDATE_SHADER_DIR(可修) |

→ **推翻了"from-scratch vulkan 无法真机测"的结论**:经 oracle-runner-on-android 路径,8/10 vulkan kernel 在手机 GPU 上正确运行 + 出延迟。

## P2 — 部分 ✅ / 大件未做

- ✅ **profiler PMU 信号回喂搜索(post-hoc BD refine)**:`policy/bd.py::posthoc_bd` 实现 + 测试 + 接进 map_elites(有 device profile 时按 cache/ipc 细分 niche;host 搜索无 PMU 时 no-op)。机制就位。
- ⬜ **vulkan shader 烘焙(可 ship)**:未做(需 `.comp`→hex SPIR-V→LayerShaderType 注册,ncnn build 集成)。
- ⬜ **fp16+packing 优化器(打赢 ncnn 生产路径的前提)**:未做(大件)。当前 baseline 是 fp32;要在 fp16+packing 档位赢 ncnn,需优化器能生成 fp16/packed 变体 + 真机测。

## P3 — whole-network:未做(大件)

单算子链路已通;整网组装 + 端到端加速需新设计(算子编排成真实模型 + net 级真机测)。

---

## 本轮发现的真实 bug(值得后续修)

1. **optimize arm/vulkan evaluator 跑不了带权重/functional 算子 baseline**(`load_model failed`/`ret=-100`)→ 阻塞 arm 优化 + wiki 消融。
2. **vulkan Gemm/Gemm_alpha shader 编译失败**(既有坏 shader)。
3. **vulkan device harness 不支持 multi-shader 算子**(Greater;需传 CANDIDATE_SHADER_DIR + 推 extra .comp)。

## 结论

- **P0 完成**(真机测量 + QD>best-first 证据)、**P1 基本完成**(8/10 vulkan 真机)、**P2 部分**(profiler 回喂机制)、**P3 未做**。
- **真机数据齐**:CPU fp32 公平对比(ours 快 2–8×)+ Vulkan GPU 8 算子(延迟+正确)。
- 剩余大件(fp16 优化器 / vulkan 烘焙 / 整网)是多日工作量,已诚实标注。
