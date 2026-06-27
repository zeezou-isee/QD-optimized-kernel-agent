# 方案C（双轨版）：覆盖 arm 变体 + 适配 arm64 宿主

> **状态**：设计文档（尚未实现）。本文件取代早先"覆盖 base + park arch 变体"的单轨设计——
> 那一版在 arm64 真机上实际跑的是标量 base，测不到真实移动算子性能，且失去了与 ncnn
> NEON kernel 的对比参照。本版改为**覆盖 arm 变体**，并把宿主从 x86 切到 arm64（Mac）。

---

## 1. 背景与动机

### 1.1 旧单轨设计的问题
旧方案C 把 agent kernel 同名覆盖进 `src/layer/<name>.cpp`，并把所有 arch 变体
（`x86/<name>_x86.cpp`、`arm/<name>_arm.cpp` …）改名挪走（park），强制 registry 回退到
base creator。这在 **x86 上测性能**时成立，但对"移动算子优化"的真实目标不成立：

- **arm64 真机跑的是 arm 变体，不是 base。** registry 把 `Softmax` 映射到 `Softmax_arm`
  的 creator（`Softmax_arm : public Softmax`，只重写 `forward_inplace`，开
  `support_packing=true` + NEON + NC4HW4）。把 arm 变体 park 掉、强制跑标量 base，
  测到的数字不代表部署形态。
- **失去对比参照。** ncnn 手调的 NEON kernel 被移出对比，"agent kernel 比 ncnn 快"
  失去被比较对象。

### 1.2 切到 arm64 宿主（Mac）带来的关键变化
开发机与部署机**同为 arm64/NEON**，于是：

- agent 的 `<name>_arm.cpp`（NEON intrinsics）可在**本机原生编译**（x86 上编不过，
  因为 `#include <arm_neon.h>` / `float32x4_t`）。
- 正确性 `allclose` 是数值等价问题，任一 arm64 机器结论一致 → **放本机最快闭环**。
- arm 变体 `: public <Native>` **继承原生 base 的 `load_param`**，因此模型 `.param`
  按原生 param-id 喂参由原生 base 解析——**param 契约对齐问题自动消失**，不再需要
  prompt 强约束 + allclose 兜底去抓。

---

## 2. 双轨职责划分

| 阶段 | 在哪做 | 覆盖目标 | 目的 |
|---|---|---|---|
| **A. 正确性轨** | PC 端（Mac, arm64 宿主） | 本机 LayerOracle 直接编译候选 `_arm.cpp`（packing=4） | 编译 + 对 PyTorch allclose（数值等价闸门） |
| **B. 性能轨** | 移动端 arm64 真机 | 覆盖 `src/layer/arm/<name>_arm.{h,cpp}` → 交叉编译 `build-android-aarch64` → adb push benchncnn | 真实 SoC 上测时序，和 ncnn 原生 NEON kernel 对拍 |

要点：
- **base 保持原生不动**（既是 param 解析来源，也是 x86 上仍可用的标量参照）。
- arm 变体是**覆盖（继承原生 base），不是 park**。
- 正确性在 A 轨用本机 LayerOracle arm 路径（已存在：`KernelAgent(backend="arm")` +
  `--packing 4` + `extra_includes=src/layer/arm`）即可，无需上真机。
- 性能在 B 轨用现有 `production_validation.py` 的 `build-android-aarch64` + `adb` 流程。

---

## 3. ncnn 机制（已核实，决定实现细节）

1. **arm64 调度**：`create_layer_cpu` 先取 `layer_registry_arch[index]`（arm 变体
   creator），非空就用它；为空才回退 base（layer.cpp:507-510）。所以"让 agent kernel
   在真机跑"= 覆盖 arm 变体的源码（保留其 creator 注册）。
2. **arm 变体形态**：`Softmax_arm : public Softmax`（softmax_arm.h），构造函数
   `support_packing = true`，只重写 `forward_inplace`；NEON + NC4HW4 packed
   （`float32x4_t` / `vld1q_f32` …）。
3. **fp16/bf16 子变体是独立手写源**：`softmax_arm_asimdhp.cpp` 单独定义
   `Softmax_arm::forward_inplace_fp16s`（softmax_arm_asimdhp.cpp:779），由
   `softmax_arm.cpp:448` 在 `opt.use_fp16_storage && elembits==16` 时分派；CMake 用
   `ncnn_add_arch_opt_source(... asimdhp ...)` 把它按 fp16 编译选项再编一遍。
   **含义**：若只覆盖 `softmax_arm.cpp`、`.h` 不再声明 `forward_inplace_fp16s`，则
   `asimdhp.cpp` 里那个成员定义会变成"定义了一个未声明的成员"→ 编译失败。
   → **覆盖 arm 变体时必须同时 park 掉 `<name>_arm_*.cpp`（fp16/bf16/dotprod 等子变体）**，
   并让 agent 的 `_arm.h` 只声明它实现的 forward。子变体被 park 后，
   `ncnn_add_arch_opt_source` 因文件不存在而跳过，fp32 NEON 路径正常工作。

---

## 4. 安全回滚（核心要求不变）

沿用并扩展现有 `NativeOverrideHandle`（net_oracle.py）：

- `overwritten`: path → 原文（`src/layer/arm/<name>_arm.{h,cpp}` 覆盖前内容）
- `parked_arch`: 被挪走的 fp16/bf16 子变体 `(orig, parked)`
- `created`: 覆盖时若某文件原不存在则记为新建（删除即可）

`restore_native_override()`：移回 parked 子变体 → 还原被覆盖的 `_arm.{h,cpp}` → 删新建 →
（因增删文件）`reconfigure_and_rebuild`。全程 try/finally、best-effort，单步失败继续其余。

**调用约定不变**：在 `OperatorAgent.run()` 的 `finally` 块**无条件**恢复，确保异常/优化失败
时下一个算子开始前 ncnn 树为干净原生状态；批量脚本每个算子之间因此天然隔离。

---

## 5. 与旧实现的差异（要改的点）

> 以下仅为设计；本次不动代码。

1. **`net_oracle.install_native_override` 增加 `target="base"|"arm"` 参数**：
   - `target="arm"`：覆盖目标改为 `src/layer/arm/<name>_arm.{h,cpp}`；类名重写为
     `<Native>_arm`（而非 `<Native>`）；**park `src/layer/arm/<name>_arm_*.cpp`**
     （fp16/bf16/dotprod 子变体），**不动 base**，**不动 x86 变体**。
   - `target="base"`：保留旧行为（x86 上测性能/可移植 kernel 时仍可用）。
   - 类名重写规则：`Cand_<Op>_arm` → `<Native>_arm`；头文件保护宏、`#include "<base>.h"`
     保持指向原生 base 头（arm 变体要 include 原生 base 的 `.h`）。
2. **`reconfigure_and_rebuild`**：arm 轨的本机正确性其实走 LayerOracle（不重建整个
   libncnn）；只有性能轨的真机交叉编译需要 reconfigure。区分清楚：
   - A 轨正确性：`LayerOracle(backend=arm, packing=4)` 直接编候选 + base extra_source。
   - B 轨性能：覆盖 arm 变体后,`build-android-aarch64` 交叉编译 + benchncnn。
3. **`OperatorAgent`**：
   - native-override 分支对"已原生 + 单层"算子,改为驱动 `KernelAgent(backend="arm",
     base_kernel_code=原生base, base_profile=...)` 写 NEON kernel；正确性走 arm LayerOracle。
   - 覆盖 `src/layer/arm/<name>_arm.*` → （真机）交叉编译 + benchmark；→ 恢复。
   - 优化 [6] 复用现有 `backend="arm"` 的 OptimizeAgent 路径（它本就把 base `.cpp` 当
     fixed extra source、NC4HW4 packing 编译候选）。
4. **宿主要求**：A 轨必须在 arm64 宿主（Mac/arm Linux）运行；x86 上 arm 轨不可用
   （README 已写明 `arm backend requires an arm64 host`）。CLI/脚本在非 arm64 宿主上
   应给出明确报错而非静默退化。

---

## 6. KernelAgent 在 arm 覆盖模式下的输入

- `backend="arm"`，`base_kernel_code` = **原生 base 的 `<name>.{h,cpp}`**（让 arm 子类
  正确 `#include` 并继承 `load_param`/成员），`base_profile` 由原生层 param 契约推导。
- prompt（`ARM_LAYER_BACKGROUND` 已具备）：要求 `class <Cand>_arm : public <Native>`，
  开 `support_packing`，只重写 `forward_inplace`，NEON 4-wide + 正确标量尾，
  elempack==1 回退路径与 base 数值一致。
- **不要求** agent 重写 param 解析（继承 base）——这是相对旧 base-覆盖方案的最大简化。

---

## 7. 验证策略

- **A 轨（本机, Mac）**：`KernelAgent(backend="arm")` 编译候选 + LayerOracle packing=4
  对 PyTorch allclose。注意本机 LayerOracle 关 fp16/多线程（fp32 NEON）；若 kernel 在
  fp16 下有精度差异，需在真机端到端数值再校验一次。
- **B 轨（真机）**：覆盖 arm 变体 → `build-android-aarch64` → adb push benchncnn →
  解析 min/max/avg；与原生 arm kernel 同条件对比。
- **回滚断言**：每个算子跑完后 `src/layer/arm/<name>_arm.*` 文本 == 原生、子变体都在原位；
  连续跑两个算子验证隔离（沿用 `eval/overwrite_native_batch.py` 的断言，覆盖目标改 arm）。
- **无依赖自检**：`eval/test_native_override.py` 增加 arm-target 的 install/restore 往返
  （临时假 ncnn 树：`arm/<name>_arm.{h,cpp}` + `arm/<name>_arm_asimdhp.cpp` 子变体），
  断言覆盖 + park + 还原后字节级一致。

---

## 8. 边界与风险

- **多层分解算子**（LogSoftmax→Softmax+UnaryOp）：仍跳过，summary 标 N/A（不变）。
- **fp16 精度**：本机 fp32 allclose 不覆盖真机 fp16 路径；要么 park 掉 fp16 子变体让真机
  也走 fp32 NEON（最稳，但不测 fp16 性能），要么真机补 fp16 数值校验。建议默认 park
  fp16 子变体（与"只优化 fp32 NEON kernel"目标一致）。
- **宿主架构**：x86 上 A 轨不可用——必须 Mac/arm64。脚本启动时检测 `platform.machine()`，
  非 arm64 直接报错并提示用 base 轨或换宿主。
- **Mac SoC ≠ 目标手机 SoC**：正确性可本机定，性能结论以真机为准。
- **回滚面**：仅触碰 `src/layer/arm/<name>_arm.*`（覆盖）+ 同名子变体（park）；base 与
  x86 变体不动，污染面比旧方案更小。
