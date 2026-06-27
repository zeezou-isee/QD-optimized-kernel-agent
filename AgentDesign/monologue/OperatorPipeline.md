# OperatorPipeline — 三后端编译与验证设计总结

## TL;DR

LLM 生成 ncnn 算子（base/arm/vulkan）后，本文讲清"怎么验证它编得过、算得对、跑得快"，核心一条贯穿始终：**验证时跑的必须是"自己生成的、指定后端"的算子，不能是 ncnn 内建或退化版**。

- **编译**：两道独立的门——`oracle` 单文件链接现成 `libncnn.a`（快）、bridge 走 ncnn CMake 重编库（真），前者过不代表后者过。当前 **base ✅ / arm ✅（需 arm64 host）/ vulkan ✅（隔离实例化 harness 已打通，宿主 GPU 经 MoltenVK 验过）**。
- **正确性**：一律 `new Cand_<Op>...` **直接实例化** + 对拍 PyTorch，绕开 `create_layer`/注册表。**自生成**三后端都保证；**非退化**：base 天然、vulkan（forward 签名不同 + `support_vulkan` 断言）、arm（`arm_forward_overridden` 静态检查）均已加固。
- **vulkan 设备**：正确性用**宿主 GPU 即可**，移动设备只在**性能**阶段才需要。
- **性能**：迭代用**隔离微基准**（已是宿主主路）、终判用 **benchncnn**；单算子真机性能**不必集成进框架**、**可跳过 benchncnn**。
- **测自己的算子（关键）**：注册表按层类型名寻址，用**唯一命名 `Cand_<Op>` + `.param` 定向**（`retarget_param_output_layer`）确定性锁定到自己的实现，**不删 ncnn 内建**。定向已接入**三处 net 级验证**：[4] `_net_numeric`、[5] `production_correctness`、benchmark。
- vulkan 实现细节见姊妹文档 [vulkan-verification-harness.md](./vulkan-verification-harness.md)。

---

> 本文总结 KernelAgent 在 **base / arm / vulkan** 三种后端下的"编译验证"链路：
> 算子怎么验证能否编译、当前模式支持哪些后端、集成进 ncnn 与 oracle 验证的差异、
> 以及对一个**已存在算子**新增 vulkan 实现时"覆盖 vs 附加 / 防冲突 / 防调错后端"的机制。
> 所有结论均基于本仓库 `opgen/` 与隔壁 `ncnn/` 源码核对，非凭记忆。

---

## 0. 两条编译路径（贯穿全文的前提）

系统里存在**两条独立的编译路径**，"oracle 编过" ≠ "集成进框架一定编过"：

| | oracle 编译（快速验证门） | 集成进框架（bridge，stage [2]） |
|---|---|---|
| 实现位置 | `opgen/layer_oracle/oracle.py: LayerOracle.compile()` | `opgen/layer_oracle/net_oracle.py: install_layer() + rebuild_libncnn()` |
| 谁编译候选 .cpp | 手搓 `g++ -std=c++11 -O2` | ncnn 自己的 CMake 构建系统 |
| 链接对象 | candidate.cpp + runner.cpp + **现成** `libncnn.a`（库里**没有**该算子） | 把 .cpp 拷进 `src/layer/`，`cmake --build` **重编整个** `libncnn.a`（算子在库里） |
| 改 CMake / 注册 | ❌ 不碰 | ✅ 往 `src/CMakeLists.txt` 插 `ncnn_add_layer(Class)` |
| `DEFINE_LAYER_CREATOR` | 剥掉（runner 直接 `new`，creator 是死代码） | 剥掉（由宏自动生成，避免重复符号） |
| 头文件定位 | `-DCANDIDATE_HEADER="xxx.h"` 显式指定 | `ncnn_add_layer` 按**类名小写**找 `cand_xxx.h` |

**为什么两条都需要**：oracle 是秒级、不重编库的快速门；集成编译多做了 `ncnn_add_layer` 代码生成、不同的 CMake flag、注册名校验等 oracle 不做的事，这些可能引入 oracle 抓不到的新失败。因此 `operator_agent` 把集成编译作为独立的 stage [2]，结果落在 `phases.install.libncnn_rebuilt` 与 `operator/libncnn_rebuild.log`，**不假设 oracle 通过即集成通过**。

---

## 1. 算子怎么验证"能否编译成功"

编译这一关由 `LayerOracle.compile()` 实现，本质是**一次真实的 `g++` 调用**（非静态检查）：

```
g++ -std=c++11 -O2 \
    -I <candidate目录> -I ncnn/src -I ncnn/src/layer -I build_lib/src \
    layer_oracle_runner.cpp  <candidate>.cpp  [+ base.cpp(arm用)] \
    build_lib/src/libncnn.a \
    -DCANDIDATE_HEADER="xxx.h" -DCANDIDATE_CLASS=Cand_Xxx -o runner
```

- `returncode != 0` → 抛 `RuntimeError` 带完整日志 → 这就是 **compile 失败信号**，喂给 debugger 角色修。
- 编译成功后才进入 run/numeric 对拍。历史 runs 里的 `"compile": true` 即此关结果。
- runner（`layer_oracle_runner.cpp`）用 `new CANDIDATE_CLASS()` 直接实例化，按 `one_blob_only/support_inplace` 分派 **CPU 的 `forward(Mat&, Mat&, opt)` / `forward_inplace`**。
- 有 mtime 缓存：候选文件没改就不重编。

---

## 2. 三后端编译支持现状

| backend | 能编译验证吗 | 怎么做 | 关键约束 |
|---|---|---|---|
| **base** | ✅ 完整 | 上面那条 g++，opt 全关、elempack=1 | 任意 host 都可 |
| **arm** | ✅ 但有前提 | 额外把已验证的 base `.cpp` 作 `extra_source` 编进来、`-I src/layer/arm`（拿 `neon_mathfun.h`），运行时 `--packing 4` 走 NC4HW4 | **NEON 仅在 arm64 host 上真正生效**（代码 `#if __ARM_NEON`；x86 上能"编过"但 NEON 分支是死代码，等于没验到 arm）；单线程、无 fp16 |
| **vulkan** | ✅ 生成 + 验证已打通（隔离实例化） | agent 写三件套（`VULKAN_LAYER_BACKGROUND` prompt）→ `VulkanLayerOracle`：`find_package(ncnn)` 链接带 vulkan 的 `libncnn`，`new Cand_<Op>_vulkan()` 直接实例化在 GPU 上跑 `forward(VkMat,...)`；shader 用独立 `.comp`、运行时 `compile_spirv_module` 编译 | 运行需 Vulkan 设备（**宿主 GPU 经 MoltenVK 即可**，无设备时优雅 SKIP）；**仅 bridge 永久安装 `.comp` 尚缺**（仅最终 register/整网用） |

**arm 的本质**：不是交叉编译到 ARM，而是"在 arm64 宿主机上原生编译"（NEON 是 arm64 baseline，无需 `-march`）。所以 "arm requires an arm64 host" 是硬约束。

> **vulkan 现状（2026-06 更新）**：第 3 节原本论证"光开 `NCNN_VULKAN=ON` 不够、缺三处机器"——其中**验证骨干（oracle/runner/bridge-shader 之外的隔离路径）已实现并验过**（手写 `Cand_AbsVal_vulkan` 在 Apple M5 上 `max_diff=0`）。第 3 节保留作为"为什么难 + 每个缺口怎么补"的分析，结论见 3.5；完整实现细节见 [vulkan-verification-harness.md](./vulkan-verification-harness.md)。

---

## 3. 打开 `-DNCNN_VULKAN=ON` 能解决 vulkan 编译吗？（分析 → 已实现）

> **状态**：本节最初论证"光开开关不够"。现在**验证骨干已落地**（见 3.5），结论是：开关是第 0 步，3.2 列的三处缺口里**oracle 验证端已用"运行时编 shader + 隔离实例化"打通**，bridge 的 shader 安装与"生成端"仍是后续。下面保留原分析以解释每个缺口及其解法。

**开关只是第 0 步，远不充分。** 它只让 libncnn "有" vulkan 运行时和 shader 构建宏；最初"生成→oracle 验证→bridge 集成"整套机器是按 **CPU layer 模型**写的，而 vulkan layer 结构上完全不同。

### 3.1 vulkan layer 不是"一个自包含 .cpp"，而是三件套

以真实的 `ncnn/src/layer/vulkan/sigmoid_vulkan.*` 为例：

- `xxx_vulkan.{h,cpp}` —— 只是 C++ 胶水：`create_pipeline` / `destroy_pipeline` / `forward_inplace(VkMat&, VkCompute&, opt)`。
- `shader/xxx.comp`（+ `xxx_pack4.comp` 等变体）—— **真正的算法在 GLSL compute shader 里**，不在 .cpp。
- `.cpp` 只负责建 `Pipeline`、设 specialization 常量、绑定 binding、dispatch。

shader 由专用 CMake 宏 `ncnn_add_shader` 预处理成 `.comp.hex.h`，在**构建期烘进 libncnn**，生成 `LayerShaderType::xxx` 枚举。

### 3.2 为什么开关救不了三处机器

1. **生成端**：`kernel_prompts.py` 只有 base/arm 背景，`_background()` 非 arm 即 base；CLI 只认 base/arm。**没有任何代码生成 `_vulkan.{h,cpp}` + `.comp`**。
2. **oracle 验证端（最根本）**：`layer_oracle_runner.cpp` 是 `new CANDIDATE_CLASS()` 调 CPU `forward(Mat&, Mat&, opt)`；vulkan 入口是 `forward_inplace(VkMat&, VkCompute&, opt)`，签名都对不上，且 `opt.use_vulkan_compute=false` 写死。要跑 vulkan 需建 `VulkanDevice`、Mat↔VkMat 上传下载、`VkCompute` 命令缓冲、`create_pipeline` 编 SPIR-V——runner 全无。
   更要命：oracle 的核心技巧是"**候选 .cpp 当独立目标，链接现成 libncnn.a**"。但 vulkan .cpp 引用的 `LayerShaderType::xxx` **只在构建期烘 shader 时才生成**，候选 .cpp **无法独立链接**——oracle 这套"不重编库、单文件验证"模式对 vulkan 直接失效。
3. **bridge 集成端**：`install_layer` 只写 .cpp/.h 到 `src/layer[/subdir]`，`rebuild_libncnn` 只做 `cmake --build`（增量）。vulkan 还需把 `.comp` 放进 `src/layer/vulkan/shader/` 并 **reconfigure** 触发 `ncnn_add_layer.cmake` 里 `file(GLOB layer/vulkan/shader/${name}.comp)` + `ncnn_add_shader`。只 `--build` 不 reconfigure，新 `.comp` 不会被 GLOB 到。

### 3.3 硬性环境前提

**已验证更正**：`NCNN_SIMPLEVK=ON`（默认）自带 vulkan 头 + 运行时 dlopen，glslang 是 bundled submodule——所以**编译 `NCNN_VULKAN=ON` 的 libncnn 不需要 Vulkan SDK**（只需 `git submodule update --init glslang`）。**运行期**才需真实 GPU + Vulkan ICD（macOS 用 MoltenVK；ncnn simplevk 读 `NCNN_VULKAN_DRIVER` 直接 dlopen，无需完整 loader）。README 把 vulkan 标 deferred 是因当时无 GPU 设备，现宿主 GPU 已可用。

### 3.4 让 vulkan 真正跑通所需（按依赖顺序，含落地状态）

1. ✅ 用 `-DNCNN_VULKAN=ON`（带 glslang）重编 libncnn（`ncnn/build_lib_vk` + install 前缀）。
2. ✅ **生成端**：KernelAgent 已加 vulkan 通路——`VULKAN_LAYER_BACKGROUND` prompt 教 LLM 产出 `_vulkan.{h,cpp}` + `.comp` 三件套；profile 由 base `as_backend("vulkan")` 派生（含 `.comp` 文件名）；`verify_kernel` 按 backend 路由到 `VulkanLayerOracle`（传 `shader=`，无 GPU 时优雅 SKIP）；CLI `--backend vulkan`。详见 [vulkan-verification-harness.md](./vulkan-verification-harness.md) §9。
3. ✅ **新的 vulkan oracle**：`vulkan_oracle_runner.cpp`（建 device、VkMat I/O、VkCompute、create_pipeline、dispatch）+ `VulkanLayerOracle`。**关键转折**：用 `compile_spirv_module` **运行时**编 shader（独立 `.comp`，经 `-DCANDIDATE_SHADER` 注入），绕开"shader 必须构建期烘入"的死结，从而**仍可隔离实例化**、用 `find_package(ncnn)` 链接 install 包。
4. ⏳ **bridge 改造**：支持安装 `.comp` 到 shader 目录并强制 cmake reconfigure（永久注册路径）。**未做**——隔离验证不需要它（运行时编 shader），仅最终 register/整网集成需要。
5. ✅ GPU 主机做 correctness：**宿主 Apple M5 经 MoltenVK** 已验 `max_diff=0`；perf 仍需真机。

### 3.5 结论（现状）

- **开关 + 验证骨干 + 生成端**：已打通——agent 可端到端写 vulkan 三件套，单算子 kernel **隔离实例化**在 GPU 上验正确性（手写样例 `max_diff=0`；agent 通路 prompt/路由/CLI 已接）。
- **核心解法**：把"shader 构建期烘入"换成"**运行时 `compile_spirv_module`**"，使 vulkan 也能像 base/arm 一样"单候选 + 隔离实例化"，且 forward 签名不同 + `support_vulkan` 断言 → 不可静默退化（见第 6 节）。
- **仍缺**：bridge 永久安装 `.comp` + cmake reconfigure（仅"永久 register/整网集成"需要；隔离验证用运行时编 shader，不需要它）。

> 一句话（更新）：开关只解决"库里有没有 vulkan 运行时"；**"能不能验" + "agent 会不会写" 都已解决**；只剩"bridge 永久装 `.comp`"用于最终 register/整网。

---

## 4. ncnn 多后端注册与运行时选择机制（理解第 5 节的基础）

核对 `src/layer.cpp` 与 `cmake/ncnn_add_layer.cmake`：

每个算子类型只占**一个 index**，但有**三张并行注册表**，同一 index 各放一个 creator：

| 表 | 内容 | 符号示例 |
|---|---|---|
| `layer_registry[i]` | base/CPU creator | `Sigmoid_layer_creator` |
| `layer_registry_arch[i]`（+avx/rvv…运行时分发变体） | arch creator | `Sigmoid_arm_layer_creator` |
| `layer_registry_vulkan[i]` | vulkan creator | `Sigmoid_vulkan_layer_creator` |

- `create_layer(i)` 建一个 **`Layer_final` 复合层**，同时持 `layer_cpu`（`create_layer_cpu` 按运行时 CPU 特性在 arch/base 间选）与 `layer_vulkan`（`create_layer_vulkan`）。
- 运行时在 `load_param`/`create_pipeline` 决定（`layer.cpp` 约 275–296）：

  > **当且仅当 `vkdev != 0` 且 `layer_vulkan->support_vulkan == true`** 才用 vulkan；
  > 否则 **`delete layer_vulkan`，静默回退到 `layer_cpu`**。

- ncnn 还提供 `create_layer_vulkan(type)`（`layer.cpp:522`）：**直接拿 vulkan creator 实例化，绕过复合层与回退逻辑**。

**这条"静默回退"是 vulkan 验证的命门**（见 5.3）。

---

## 5. 对"已存在算子"新增 vulkan 实现：覆盖/附加、防冲突、防调错

### 5.1 覆盖还是附加？→ **附加（同 index、另一张表的空槽）**

- 新增**独立文件** `src/layer/vulkan/<name>_vulkan.{h,cpp}` + `shader/<name>.comp`，**完全不动** base/arm 的 .cpp。
- vulkan 类**继承现有 base 类**（`class Sigmoid_vulkan : public Sigmoid`）。
- 该算子的 `ncnn_add_layer(Sigmoid)` 已存在；`NCNN_VULKAN=ON` 且 `WITH_LAYER_<name>_vulkan` 时宏自动 GLOB vulkan 文件与 shader，把原本 `{"Sigmoid",0}`（空）的 `layer_registry_vulkan[i]` 填上。
- 结论：**严格附加**——补 vulkan 空槽，base/arm 槽原样不动。

### 5.2 怎么保证不冲突？

ncnn 这侧**天然不冲突**：
- **名字不重**：注册名仍是 `Sigmoid`（一名一 index），不是注册第二个 "Sigmoid"；三个 creator 在三张数组同一下标，结构上不可能撞。
- **符号不重**：`Sigmoid_vulkan_layer_creator` 与 base/arm creator 天然区分；`DEFINE_LAYER_CREATOR(Sigmoid_vulkan)` 由宏生成——**绝不能自己写**（与 base/arm 同规则，bridge 已 strip）。

**风险在 agent/bridge 侧**：
1. 已存在算子**不能再插一行 `ncnn_add_layer(Sigmoid)`**（会重复）。加 vulkan 时应**完全不碰 ncnn_add_layer**，只丢文件 + cmake **reconfigure** 重新 GLOB。
2. **shader 文件名必须等于 layer 文件 stem**（`file(GLOB shader/${name}.comp)`）；不匹配不会报冲突，而是**静默没编进去**——比冲突更难查。

### 5.3 验证时怎么保证不调用错原来的算子？

因 `Layer_final` 静默回退：若**没装 vkdev** / vulkan 类**忘了 `support_vulkan=true`** / `create_pipeline` 失败 → ncnn **悄悄删 vulkan 层、用回原 CPU 算子**，而 allclose 照样绿——你以为验了 vulkan，其实验的是原版 CPU 实现。这是"调错算子"的核心陷阱。

防住它（建议两者都用）：

**(a) 隔离验证（首选，与现有 oracle 同思路）**
用 `create_layer_vulkan(type)` **直接实例化 vulkan creator，绕过 `create_layer`/`Layer_final` 的复合+回退**；再配合**候选独立类名**（`Cand_Sigmoid_vulkan : public Cand_Sigmoid`，而非真的 `Sigmoid`），从根上不绑到内建实现。等价于现 `LayerOracle` 的 `new CANDIDATE_CLASS()`，没有 fallback 就不可能调到原算子。

**(b) 整网验证时显式断言**
若必须走 Net（NetOracle）：
1. 必须 `opt.use_vulkan_compute=true` 且 `net.set_vulkan_device(...)` 装真实 `VulkanDevice`，否则 vulkan 永不被选。
2. load 后**断言后端**：检查该层 `support_vulkan==true` / `layer_vulkan` 未被删，**绝不能只凭 allclose 通过就认为跑了 GPU**。
3. 性能尤其危险：静默回退到 CPU 会给出"看着正常"的延迟，把优化结论带偏。

---

## 6. 正确性验证的隔离保证（验证时跑的一定是"自己生成的、指定后端"算子）

验证一律是 `oracle.run` 里 `new CANDIDATE_CLASS()` **直接实例化** + allclose，**不经过 `create_layer`/注册表/`Layer_final`**。这给两层保证，但三后端强度需对齐：

- **保证 1：一定是自生成、不撞 ncnn 内建** ✅（三后端都满足）。类名 `Cand_<Op>[_arm|_vulkan]` 唯一，直接 `new`，不查注册表。
- **保证 2：是指定后端、非退化**：
  - **base**：本身即 base，无退化问题。
  - **vulkan** ✅ 不可退化：vulkan 的 `forward(VkMat&, VkCompute&)` 与 CPU `forward(Mat&)` **签名不同**，不 override 则 base 默认 `forward_inplace(VkMat)` 返回 -1 → runner 失败；再加 `support_vulkan==true` 断言（否则 rc=6）+ 不走 `create_layer`。
  - **arm** ✅ 已加固：arm 子类的 `forward(Mat&)` 与 base **同签名**，不 override 会**静默回落到继承的 base CPU forward**，对 elementwise 算子 numeric 仍 PASS（假阳性）。加固方式——`kernel_pipeline.arm_forward_overridden()` 静态检查生成代码必须定义 `<arm_class>::forward[_inplace]`，否则即使 numeric 通过也**翻成失败**喂给修复循环；arm 生成 prompt 也声明该 override 为强制项。

> 一句话：自生成三后端保证；非退化 base 天然 / vulkan 与 arm 均已加固。

---

## 7. vulkan 正确性验证需要接移动设备吗？→ 不需要

正确性需要的是**一个 Vulkan 设备**，但**宿主机自己的 GPU 即可**（本机 Apple M5 经 MoltenVK 已验 `max_diff=0`），不必接真机。分层与 CPU 一致：

| 阶段 | 设备 | 理由 |
|---|---|---|
| 正确性（allclose） | 任意 Vulkan GPU，**宿主即可** | compute shader 数值在符合规范实现间可移植（fp 容差内），宿主 GPU 是有效 oracle |
| 性能（延迟） | **真实目标设备**（移动 Adreno/Mali） | 延迟硬件相关，宿主 GPU 耗时 ≠ 移动 GPU |

caveat：MoltenVK（Metal 后端）与移动真机驱动在边角行为（fp16 舍入/subgroup）可能不完全一致——宿主 GPU 正确性是**强信号非终判**，最终真机做 perf 时顺带再过一遍正确性确认。

---

## 8. 性能验证：要不要集成进框架 / 能不能跳过 benchncnn

### 8.1 现状：性能有两条路，主路已是隔离实例化

| 路径 | 在哪测 | 怎么测 | 集成进框架 | 用 benchncnn |
|---|---|---|---|---|
| **optimizer measure**（`optimize/evaluator/measure_harness`+`cpu_runner`）| 宿主机 | 复用 LayerOracle 隔离 runner，warmup+N 次取 min/median/std | ❌ | ❌ |
| **production benchmark**（`production_validation.benchmark`）| 真机 | 交叉编 benchncnn → adb → 跑整张 .ncnn 模型 | ✅ | ✅ |

即驱动 QD 优化的性能信号**已经是"单算子 + 隔离 + 宿主机"**；只有最终真机 benchmark 用 benchncnn 且需整网。

### 8.2 单算子真机性能要集成进框架吗？→ 不必

同隔离思路：把宿主 runner 交叉编到 android arm64 + adb 跑即可，不需要 `ncnn_add_layer`/整网转换。vulkan 同理（runner 用 `NCNN_VULKAN=ON` 交叉编，跑真机 Adreno/Mali）。

### 8.3 能跳过 benchncnn 吗？→ 能，但要自己接管"保真旋钮"

benchncnn 不是硬依赖。自带计时 runner 需补齐：**进程内计时**（不能 wall-clock 套 subprocess，现 `cpu_runner` 注释自承这是 M1 局限）、**大核亲和/powersave**、**线程数 + OMP 池预热**、**真实 elempack 布局**、**控热**。诚实结论：这些机械旋钮可复刻，但**整网上下文相关的量（层间 repack、内存池热度）无法逐位复现**。

→ 分层（与正确性对称）：**迭代内循环用隔离微基准**（只需一致性，不需 benchncnn 级保真）；**最终判定用 benchncnn**（整网终判）。

### 8.4 用 benchncnn 时如何保证"测的是自己的算子"——`.param` 定向（已实现并接入）

注册表按**层类型名**寻址。我们的算子注册名是 `Cand_<Op>`（唯一），ncnn 内建是 `<Op>`，**天然不同名**。benchncnn 用哪个，**只看 `.param` 里写的类型名**：

- **新算子**：GraphAgent 以 `force_target_layer=cls` 转换 → .param 已写 `Cand_<Op>`。
- **已存在算子**（`already_in_ncnn=True`）：走 ncnn 原生转换 → .param 写**内建名**（且原生名常**无法**从任务名推出，如 `torch.exp`→ncnn `UnaryOp`）→ 默认 benchmark 跑的是内建实现。**这是缺口。**

**解法（已落地）**：`net_oracle.retarget_param_output_layer(param_text, "Cand_<Op>")` ——把**产出最终输出 blob 的那一层**的类型改写成 `Cand_<Op>`（对单算子参考模型即是被测算子，绕开原生名不可推的问题）；只改类型 token，blob/层名/params/空白全保留；对新算子是 `cls→cls` 幂等空操作。

**接入点**：`production_validation.benchmark(param, shapes, retarget_to=cls)` 在 adb push 前重定向并 push 重写后的 `.param`（benchncnn 只需 param、用空权重，故无需 .bin）；编排器 `_run_production_step(..., op_class=cls)` 在 [5] production 与 [6] 优化复验两处都传入 `cls`。结果里带 `retargeted` 计数。**全程不删/不改 ncnn 内建**。

> 抉择：**不要原地替换/删除 ncnn 内建**（繁琐、失去 A/B、易误伤）；用**唯一命名 + .param 定向**即可确定性锁定到自己的实现。

### 8.5 net 级正确性也已接入同一定向（闭合）

`already_in_ncnn` 时，**整网正确性**（stage [4] `_net_numeric`）与 **production correctness** 之前按内建名跑，现已用同一定向闭合：两处在 `run_net` 前调 `retarget_param_output_file(param, rp, cls)`，对原始 .param 产出重写副本再跑，使整网/production 正确性也确定性地验**我们的 `Cand_<Op>`**。编排器三处统一传 `op_class=cls`：[4] `_net_numeric`、[5] `production_correctness`、benchmark。新算子幂等（cls→cls）。单算子 `LayerOracle` 走直接实例化，本就不受影响。

> caveat：net 路径用**重写后的 .param + 原始 .bin**。weightless 的 elementwise/activation（这正是"已存在算子重写"的主力场景）无影响；极少数"已存在且带权重"的算子若 `Cand_` 的 `load_model` 权重布局与原生不一致，会表现为**真实的对拍失败**（而非假阳性），仍是更诚实的信号。带权重的硬算子通常走 GraphAgent（`force_target` 已是 `Cand_`），不经此路径。

---

## 9. 一页速查

- **编译验证 = 真实 g++ 调用**，returncode≠0 即失败；oracle 单文件链接现成库，集成走 ncnn CMake 重编库——**两道独立的门**。
- **base ✅ / arm ✅（需 arm64 host）/ vulkan ✅ 生成+验证（仅 bridge 永久装 `.comp` 待补）**。
- **vulkan 验证已打通**：靠**运行时 `compile_spirv_module` 编 shader**（独立 `.comp`）+ `find_package(ncnn)` 链接 `build_lib_vk` install 包 + 隔离实例化；宿主 GPU(MoltenVK) 验 `max_diff=0`。仍缺：agent 生成三件套、bridge 永久装 `.comp`。
- vulkan layer = `_vulkan.{h,cpp}` + `.comp` **三件套**，算法在 shader 里。
- 已存在算子加 vulkan：**附加**（同 index 补 vulkan 空槽），**天然不冲突**（一名一 index、符号区分），但 bridge 别重复 `ncnn_add_layer`、shader 名须对齐。
- 验证防调错：默认会**静默回退到 CPU**；用 **`create_layer_vulkan` 隔离实例化** 或 **装 vkdev + 断言 `support_vulkan`**。
- 正确性验证 = 直接 `new Cand_<Op>...` + allclose；自生成三后端保证；非退化 base 天然 / vulkan(签名不同+断言) / arm(`arm_forward_overridden` 静态检查) 均已加固。
- vulkan 正确性用**宿主 GPU**即可（MoltenVK），移动设备只在**性能**阶段才需要。
- 性能：迭代用**隔离微基准**（已是宿主主路）、终判用 **benchncnn**；单算子真机性能**不需集成进框架**、**可跳过 benchncnn**（代价是自己接管计时/亲和/线程/布局旋钮）。
- benchncnn 测自己的算子：**唯一命名 + `.param` 定向**（`retarget_param_output_layer`），**不删 ncnn 内建**。
- `.param` 定向已接入**三处 net 级验证**：[4] `_net_numeric`、[5] `production_correctness`、benchmark（统一 `op_class/retarget_to=cls`，幂等于新算子）——已存在算子的**正确性与性能都锁定到自己的实现**。

---

## 附：关键代码位置

| 主题 | 文件:函数 |
|---|---|
| oracle 编译/运行/verify | `opgen/layer_oracle/oracle.py: LayerOracle.compile/run/verify` |
| 泛型 runner（CPU forward 分派、--packing） | `opgen/layer_oracle/layer_oracle_runner.cpp` |
| 装入树 + 重编库 + restore | `opgen/layer_oracle/net_oracle.py: install_layer/rebuild_libncnn/restore` |
| `.param` 定向（让 net 级验证测自己的算子） | `opgen/layer_oracle/net_oracle.py: retarget_param_output_layer / retarget_param_output_file / retarget_param_layer`；单测 `opgen/layer_oracle/test_retarget_param.py` |
| net 级验证定向接入点（[4]/[5]/benchmark） | `operator_agent.py: _net_numeric / _run_production_step`；`production_validation.py: production_correctness / benchmark` |
| arm 非退化静态检查 | `opgen/kernel/kernel_pipeline.py: arm_forward_overridden` |
| 集成编译/整网正确性/android bench（含 `retarget_to`） | `opgen/orchestrator/production_validation.py: ProductionValidator.benchmark` |
| 7 阶段编排（含 stage [2] BRIDGE） | `opgen/orchestrator/operator_agent.py` |
| backend 后缀占位（含 vulkan） | `opgen/kernel/kernel_schemas.py: _SUFFIX` |
| backend prompt 背景（base/arm/vulkan） | `opgen/kernel/kernel_prompts.py: _background / ARM_LAYER_BACKGROUND / VULKAN_LAYER_BACKGROUND / _files_instruction` |
| vulkan agent 通路（oracle/profile/CLI） | `kernel_agent.py`（`VulkanLayerOracle` 分派、`as_backend("vulkan")`）、`kernel_pipeline.py: verify_kernel`（shader 路由）、`kernel_schemas.py: KernelProfile.shader`、`cli/run_kernel_agent.py` |
| ncnn 多后端选择 + 静默回退 | `ncnn/src/layer.cpp: create_layer / create_layer_cpu / create_layer_vulkan / Layer_final` |
| 三张注册表 + shader 烘入 | `ncnn/cmake/ncnn_add_layer.cmake` / `ncnn/cmake/ncnn_add_shader.cmake` |
