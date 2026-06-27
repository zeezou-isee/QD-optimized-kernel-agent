# OptimizeAgent — 两层 Quality-Diversity 算子性能优化

## TL;DR

OptimizeAgent 在 KernelAgent 验证过的 kernel 上做性能优化:**外层 MAP-Elites(QD)搜 kernel 结构、内层搜 knob 参数**;每个候选都过"真值门"(编译→对拍 baseline→实测)才参选;并行一个 best-first 对照臂,用数据决定"QD 到底值不值得用"。

- **外层(QD)**:LLM 当变异算子,按**行为描述子(BD)niche** 铺开结构多样性;**局部 cell 竞争 + σ 噪声地板**保多样、反欺骗。
- **内层**:解析剪枝(免实测)→ 粗网格定位 → 爬山,在单模板里调 knob。
- **BD 维度**:2 个**结构化、生成时即可知**的轴;具体哪 2 个由 **roofline** 的访存/计算瓶颈二选一。
- **真值门**:Evaluator = materialize→compile→correctness(**对拍 baseline**)→measure(warmup+N, min/median/std);先对后快。
- **经验池(兵器谱)**:按 regime 跨算子 warm-start + 持久化。
- **本次改动(§7)**:correctness 反馈接入**失败 taxonomy**(后端感知)+ 把失败诊断**闭环回喂给 LLM proposer**。

---

## 1. 两层各管什么

| 层 | 搜索空间 | 方法 | 变异者 |
|---|---|---|---|
| **外层 MAP-Elites(QD)** | kernel **结构**(参数化模板) | 质量-多样性,按 BD niche | **LLM**(贵) |
| **内层 参数搜索** | 单模板的 **knob 取值** | 剪枝 + 粗网格 + 爬山 | 纯算法(便宜) |

外层每产一个模板,内层求其最优 knob 配置(**basin**)回填档案。

## 2. BD 维度(MAP-Elites 的轴)

每套坐标系 **2 个轴(2D 网格)**,按 roofline regime **二选一**(`bd.py`):

| regime | axis1 | axis2 | 网格 |
|---|---|---|---|
| memory_bound(A) | layout_family: nchw/nhwc/packed | tiling_strategy: none/single/double | 3×3 |
| compute_bound(B) | algo_family: direct/gemm/winograd/fft/dw | compute_mapping: scalar/vec/dotprod | 5×3 |

- **选坐标系**:`roofline.diagnose` 在"问题级"(算子+shape+硬件,naive 实现)算 `AI=flops/bytes`,与 ridge(峰值比,或默认 8.0)比 → memory/compute,判一次并锁定。
- **三原则**:① BD 答"哪一类"≠ fitness(延迟);② BD **生成时即可定位**(`classify(techniques, regime)` 对 LLM 自报标签做关键词匹配),不靠实测 → 防档案污染;③ 维度随瓶颈条件化(访存看布局/分块,计算看算法/指令映射)。
- **局限**:roofline 估计糙(`flops≈输出元素数`,对 conv/matmul 偏差大,复杂算子应显式传 `OperatorProfile`);niche 定位依赖 LLM 标签准确性;词表硬编码。

## 3. 外层 MAP-Elites 主循环(`run_map_elites`)

**冷启动**:经验池同 regime 种子(地板、不过滤)+ baseline 入档案。

**循环(预算按内层实测次数计)**:
1. **roofline 早停**:best 逼近理论地板就停。
2. **directive**:`coverage < target` → `diversify`(先铺开),够了 → `optimize`。
3. **选亲代 `select_parents`**:质量+新颖度加权采样 `score = quality(best/elite) + 0.4·novelty(1/(1+visits))`,无放回 → 好 niche 与欠探索 niche 都被选,避免塌缩到单一最快。
4. **变异 `vary_fn(parent, directive, history)`**:LLM 把亲代精英改写成新参数化模板(knob 占位 + 物理约束 + techniques)。
5. **解析预筛**:signature 去重,完全相同提案不实测。
6. **BD 预定位** `classify` → cell(内层搜索前就知落哪 niche)。
7. **内层 `inner_search`** → basin。
8. **cell 局部竞争 `archive.place(elite, σ)`**:只在**本 cell 内**比,且须快过 σ 才替换。**竞争只在格子内**→ 新颖但稍慢者占空格即存活,不被全局最优挤掉(反欺骗)。
9. 更新全局 best;`stale`/patience 收敛、预算、roofline 任一触发停。

**返回**:对所有 cell 取 `argmin`(QD 当手段,只报最快的)。

## 4. 内层参数搜索(`inner_search`)

对每个 knob 点:
- **① 解析剪枝(免实测)** `engine.feasible`:LLM 写的物理约束方程(`TILE_M*TILE_N*4<=L1` 等)+ 内建启发式(cache/寄存器/整除/对齐)在实测前砍非法点 → `n_pruned`,不评估、不可能胜出(只省预算,绝不靠估计加冕)。
- 可行 → `evaluator.evaluate`(真值门),仅 `correct` 且有延迟时更新 best。

策略:**② 粗网格**(每轴低/中/高代表值 × 笛卡尔积、封顶)定位 basin → **③ 爬山**(坐标下降 ±1 index,利用参数层局部平滑;小预算下比 TPE 便宜)。

## 5. 真值门 Evaluator(`evaluator/`)

`materialize(template, point)` → **compile**(`CpuRunner`/`LayerOracle`)→ **correctness**(`CorrectnessOracle`,**对拍 baseline**:baseline 已 == PyTorch,对拍它更便宜且输入/参数字节一致)→ **measure**(`MeasureHarness`:warmup + N 次,min/median/std 噪声地板)。任一关失败 → `MeasureSample(correct=False)`,无延迟,不参选。

## 6. best-first 对照臂 + 经验池

- **best-first(`run_best_first`)**:同预算并行的控制组——贪心、不维护档案、不保多样,每轮只把**当前最优**变快。判据:QD 更优且 argmin 来自非主流 cell → 多样性付费成功;best-first 更少轮追平 → 算子欺骗性弱,用基线。**把"用不用 QD"变成数据裁决**。
- **经验池(兵器谱,`experience_pool`)**:持久 JSON,**按 regime(非算子)** 索引;新算子用同 regime 已知 kernel 播种,跑完回灌 → 越用越富(跨算子复用)。

## 7. 本次改动(把 KernelAgent 的诊断驱动反馈搬进优化器)

**7.1 correctness 反馈升级为失败 taxonomy(后端感知)**
- `CorrectnessReport` 加 `failure_category`;`CorrectnessOracle` 失败时改调**共享 `classify_failure`**(`layer_oracle.failure_taxonomy`),带 `backend` + `input` → 自动得 E3/E4/E5/E6/**E8(vulkan 覆盖)** + **arm lane/tail** 提示。`Evaluator` 构造时传入 `backend` 和 baseline `input`。
- 每个候选的 `MeasureSample.correctness` 现在带标签+定位,而非旧标量 `max_diff`。

**7.2 闭环——把失败诊断回喂给 LLM proposer(原来是开环)**
- `map_elites._summarize_failures(basin)`:统计 basin 失败候选的**主导类别 + 代表诊断**,记进 `iters` 的 `failure_summary`。
- `proposer.vary` 从 history 抽 `failure_summary`,经新参数 `recent_failures` 注入 `vary_prompt` 的 **"Recent candidate failures (diagnosis)"** 区块 → 下一轮变异能"看着上一个模板为什么成批失败"去改(如"vectorize 变体 8/8 栽在 E6 lane[2] → 修 NEON 尾巴")。

**验证(离线,无 LLM/torch/ncnn)**:M1/M2/M3 单测全 PASS(回归);离线 e2e 确认 arm 错误→E6+lane、vulkan passthrough→E8、`_summarize_failures` 产摘要、`vary_prompt` 携带失败区块。新参数全默认 → 向后兼容。

**7.3 vulkan 优化后端(#3,已实现)**
- 新增 `VkRunner`(`evaluator/vk_runner.py`),镜像 `CpuRunner` 接口(`compile_only`/`run_once`/`read_output`)但走 `VulkanLayerOracle`(GPU 隔离实例化、`.comp` 运行时编译);`CpuRunner.compile_only` 接受被忽略的 `shader=` 以统一调用。
- `Evaluator`:`backend=="vulkan"` → 用 `VulkanLayerOracle`+`VkRunner`;`.comp` 由 `_shader_file` 识别、`_shader_path()` 每次注入;base 文件(vulkan 子类化 base)作 `extra_sources` 编入。
- `run_optimize.py` `--backend vulkan`(从 `kernel_vulkan` 载入三件套 + base 作 fixed source);`OptimizeAgent` backend 透传。
- 无 GPU 时 `run_once` 返回 skip 标记 → 优化优雅跳过(同缺设备)。
- **验证**:`test_vk_runner.py` 用样例 `Cand_AbsVal_vulkan` 经 `VkRunner` **在 GPU(MoltenVK)上编译+跑 5 次+对拍 np.abs 通过**(无 LLM/torch)——证明优化器能编译/运行/测量 vulkan kernel。

## 8. TODO / 局限

- **性能测量是宿主单算子计时**(`cpu_runner`/`vk_runner` 都是 wall-clock 套 subprocess,M1 局限)——排序够用、非真机;真机性能要接 benchmark(见 [OperatorPipeline.md](./OperatorPipeline.md) §8)。vulkan 的 84ms/次含进程启动+GPU init,需进程内计时才精确。
- basin 失败**类别分布统计**落盘、arm 非退化断言:便宜的加固/数据,未做。
- roofline 估计糙 / niche 依赖 LLM 标签:见 §2 局限。

## 附:关键代码位置

| 主题 | 位置 |
|---|---|
| 编排(Proposer/Evaluator/Policy,linear vs map_elites) | `opgen/optimize/optimize_agent.py` |
| 外层 QD 主循环 + 失败摘要 | `opgen/optimize/policy/map_elites.py: run_map_elites / _summarize_failures` |
| 档案(cell 局部竞争 / 亲代选择 / 持久化) | `opgen/optimize/policy/archive.py: Archive.place / select_parents` |
| BD 双坐标系 + roofline regime | `opgen/optimize/policy/bd.py` / `roofline.py` |
| best-first 对照臂 / 经验池 | `opgen/optimize/policy/best_first.py` / `experience_pool.py` |
| 内层(剪枝/网格/爬山) | `opgen/optimize/inner/inner_search.py / constraint_engine.py / coarse_grid.py / hill_climb.py` |
| 真值门 + 失败 taxonomy 接入 | `opgen/optimize/evaluator/evaluator.py / correctness_oracle.py / measure_harness.py / cpu_runner.py` |
| vulkan 优化 runner(#3)+ 自检 | `opgen/optimize/evaluator/vk_runner.py`;`opgen/optimize/test_vk_runner.py` |
| 三角色 prompt(含 recent_failures 闭环) | `opgen/optimize/proposer/proposer.py / prompts.py` |
| schemas(含 CorrectnessReport.failure_category) | `opgen/optimize/schemas.py` |
