# Miniset 真机测试报告(Redmi / 骁龙 778G / Adreno 642L)

> 目标:P0–P3 + miniset 全量真机结果。本报告记录**已建成的真机测量能力、拿到的真机数据、以及一个决定"公平对比"的根本发现**。所有性能数字均来自**真机**(从不 Mac 测最终性能)。

## 一、已建成:P0 真机测量后端 ✅

- `scripts/bench_miniset_device.py`:对任一生成算子,install→交叉编译 android benchncnn→retarget→真机 profile(op_profiler/simpleperf)→还原 ncnn 树。
- `op_profiler`:一次 simpleperf run 同出 PMU(IPC/cache-miss)+ 延迟;热点符号自动发现 + 可信度门控。
- 可测 native / our_base(fp32 可移植)/ our_arm(NEON)三种后端。ncnn 树每次 install 后**还原干净**。

## 二、根本发现:对比被"优化档位"和 benchncnn 自身仪表污染

跑第一版(miniset 默认 shape,benchncnn 默认 opt)时数字不可信,深挖后定位到 3 个混淆:

1. **fp16 + packing 档位不匹配(最关键)**:benchncnn 硬编码 `use_fp16_storage/packed/arithmetic=true` + `use_packing_layout=true`(benchncnn.cpp:368-373)。ncnn **原生算子跑 fp16+NC4HW4 打包的优化路径**;而**我们生成的是 fp32 标量 baseline**。两者不在同一优化档位 → head-to-head 不公平。
2. **shape 太小 → 框架主导**:miniset 默认 shape 下,很多算子 <0.05ms,净延迟被 allocator/timer 吃掉,不反映 kernel。
3. **benchncnn 自身计时仪表**:benchncnn 对每层调 `clock_gettime` 计时,tight loop 下该符号在 PMU 里占大头(已加入 denylist)。

## 三、公平的 fp32 对比结果(scale=8,fp32 benchncnn,单输入可缩放算子)

把 benchncnn 临时改成 fp32/no-packing(两边都 fp32),shape 放大 8× 让 kernel 主导。结果(真机 min ms):

| 算子 | native(fp32) | our_base(fp32) | our_arm | our/native |
|---|---|---|---|---|
| Abs | 4.63 | 2.31 | 2.26 | **0.49×(我们快 2×)** |
| BatchNormalization | 4.63 | 2.33 | 2.33 | **0.50×** |
| ReduceSum | 3.59 | 0.44 | 0.45 | **0.13×(我们快 8×)** |
| AveragePool | 0.21 | 0.24 | (arm 缺) | ~1.0× |

**解读**:在 **fp32 档位**,我们从零生成的 baseline 已经**追平甚至反超 ncnn 的 fp32 回退路径 2–8×**。原因:ncnn 的 fp32-unpacked 是未优化 fallback,ncnn 真正的速度在 fp16+packing。所以:
- ✅ 我们的 baseline 在同档位(fp32)下已很有竞争力。
- ⚠️ 要对比 ncnn 的**生产路径(fp16+packing)**,我们的 kernel 也得上 fp16+packing —— 这正是 QD 优化器该做的(目前 baseline 未做)= **P2**。

## 四、默认档位(fp16)全 11 算子数据

见 `miniset_device.json`。这版是 native(fp16)vs our(fp32),因档位不匹配 + shape 过小,数字仅供参考,不作公平结论。关键仍成立:**所有算子在真机上正确运行**(correctness 已由 host NetOracle 验证 + 真机不崩)。

## 五、P0–P3 进度与剩余

| 项 | 状态 |
|---|---|
| **P0 真机测量后端** | ✅ 建成 + 验证 + 出数 |
| **P0 fp32 公平方法学** | ✅ 确立(fp32 benchncnn + scale) |
| P0 证据实验(best-first / wiki 消融) | ⬜ 待跑(host 算法消融,不涉最终真机 perf) |
| **P1 vulkan 真机** | 🟡 单算子已证(Abs 在 Adreno GPU 上 max_diff 0.0);批量化 + 延迟仪表待做 |
| **P2 kernel 上 fp16+packing(公平对比 native 生产路径的前提)** | ⬜ 大件,未做 |
| P2 profiler 信号回喂搜索 | ⬜ |
| P2 vulkan shader 烘焙(可 ship) | ⬜ |
| P3 整网组装 + 端到端加速 | ⬜ 大件 |

## 六、需要决策的点

"我们的 kernel vs ncnn 原生"的**公平对比**取决于优化档位:
- **方案A**:同档位 fp32 对比(已出数,ours 快 2–8×)—— 证明 baseline 生成质量。
- **方案B**:让优化器把 kernel 提升到 fp16+packing,再对比 ncnn 生产路径(fp16)—— 这才是"打赢 ncnn 生产库"的主张,但需 P2(大)。

方案 B 是论文的强主张,但工作量大(优化器要能生成 fp16/packed 变体 + 真机测)。方案 A 是当下能拿到的诚实结果。
