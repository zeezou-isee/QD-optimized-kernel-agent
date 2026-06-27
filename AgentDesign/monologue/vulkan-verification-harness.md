# Vulkan 后端验证 harness（隔离实例化）— 实现细节

> 本文归纳 vulkan 后端"隔离实例化"验证通路的实现：与 base/arm 一致的设计、关键 ncnn API、
> 文件清单、踩平的 3 个真实坑、环境配置、以及已验证结果。
> 配套总览见 [AgentDesign.md](./AgentDesign.md) 的第 3、5 节。

**本期范围**：只做**验证 harness + 手写样例**，不接 agent 生成端（prompt/schema/CLI/pipeline 分发留待下一步）。
shader 形式：**独立 `.comp` 文件 + `.cpp`**。

---

## 1. 设计：为什么是"隔离实例化"

与 base/arm 的 `LayerOracle` 同一哲学——**直接 `new CANDIDATE_CLASS()`**，手动建 GPU 资源并跑
`forward(VkMat, VkCompute)`，**全程不经过 ncnn 的 `create_layer` / `Layer_final` 复合层**。

这是 vulkan 验证的命门：`Layer_final` 在 `load_param`/`create_pipeline` 里有"静默回退"逻辑——
若 `vkdev==0` 或 `layer_vulkan->support_vulkan==false` 或 `create_pipeline` 失败，就**悄悄删掉
vulkan 层、改用 CPU 实现**，而 allclose 照样绿。走复合层验证 = 假阳性陷阱。隔离实例化从根上规避它。

双重保险：
- runner 直接实例化候选 vulkan 类，没有 fallback 路径。
- 候选用**独立类名** `Cand_AbsVal_vulkan : public ncnn::Layer`（不是 ncnn 内建 `AbsVal`），且必须
  `support_vulkan=true`，否则 runner **拒绝运行**（`rc=6`）。

---

## 2. 关键 ncnn API（绕开"构建期烘 shader"的死结）

base/arm 的 oracle 把候选 `.cpp` 当独立目标链接现成 `libncnn.a`。vulkan 不能照搬，因为内建 vulkan
层引用的 `LayerShaderType::xxx` 枚举与 SPIR-V 都是**构建期**由 `ncnn_add_shader` 烘进库的，单文件
链接拿不到。破解靠两个核心 API：

| API | 位置 | 作用 |
|---|---|---|
| `compile_spirv_module(const char* comp, int size, const Option&, std::vector<uint32_t>& spirv)` | `ncnn/src/gpu.h:556`（`#if NCNN_VULKAN`） | **运行时** GLSL→SPIR-V（用 bundled glslang）。候选把 shader 写成独立 `.comp`，运行时编译 |
| `Pipeline::create(const uint32_t* spv_data, size_t size, specializations)` | `ncnn/src/pipeline.h:30` | 用裸 SPIR-V 建管线；`resolve_shader_info` 自动从 SPIR-V 推断 binding/push_constant |

`.comp` 路径由 oracle 以宏 `CANDIDATE_SHADER`（字符串字面量）注入，风格对齐已有的
`-DCANDIDATE_HEADER/-DCANDIDATE_CLASS`。

整体跑法照搬官方"隔离层 + vulkan"测试 `ncnn/tests/testutil.cpp:895-1175`：
get_gpu_device → acquire allocator → create_pipeline → record_upload → forward → record_download → submit_and_wait。

---

## 3. 文件清单

新增（`opgen/layer_oracle/`）：

| 文件 | 作用 |
|---|---|
| `vulkan_oracle_runner.cpp` | 泛型 GPU runner。复用 base runner 的 `--param/--input/--weight/--out` 与 bin 协议；`new CANDIDATE_CLASS()` → 设 `vkdev`/`use_vulkan_compute=true`/allocator → `create_pipeline` → 上传/forward/下载 → 写 `out.bin`。无 GPU 退出码 **42**；候选不 `support_vulkan` 退 6 |
| `cand_vulkan_shader.h` | inline `compile_candidate_shader(opt, spirv)`：读 `CANDIDATE_SHADER` 指向的 `.comp` → `compile_spirv_module` → 填 spirv。消除候选样板代码 |
| `vulkan_oracle.py` | `VulkanLayerOracle`：`compile/run/verify` 与 `LayerOracle` **同签名**（drop-in）。compile 生成小 `CMakeLists.txt`（`find_package(ncnn)` + 注入三个宏）→ `cmake --build`，按 mtime 缓存。run 检测 rc=42 → `OracleResult(skipped=True)`。macOS 自动探测 MoltenVK 设 `NCNN_VULKAN_DRIVER` |
| `samples/cand_absval_vulkan.{h,cpp}` + `samples/cand_absval.comp` | 手写样例算子：标量 shader（`sfp`/`buffer_ld1`/`psc`），elempack=1，`forward_inplace` |
| `run_vulkan_oracle.py` | 自检脚本：vs `np.abs`（无需 torch）。两关——COMPILE gate（任意 arm64 mac 可跑）+ RUN gate（需 MoltenVK，否则 SKIP） |

改动：
- `oracle.py`：`OracleResult` 加 `skipped: bool` 字段。
- `__init__.py`：导出 `VulkanLayerOracle`。

构建产物（在 `ncnn/`，独立仓库）：`build_lib_vk/`（`NCNN_VULKAN=ON`）+ `build_lib_vk/install/`（cmake 包）。

---

## 4. 链接策略：用 ncnn 导出的 CMake 包，不手搓 g++

vulkan 链接远比 base/arm 复杂：除 `glslang`/`SPIRV` 两个内置静态库，Apple + simplevk 还要链
`Metal/Foundation/QuartzCore/CoreGraphics/IOSurface/AppKit/IOKit` 框架、`-ldl`、`simplevk.tbd`
弱链接（见 `ncnn/src/CMakeLists.txt:276-340`）。手搓极脆弱。

解法：runner 走 ncnn 自己导出的 CMake 接口——
1. `cmake --install build_lib_vk --prefix build_lib_vk/install` 生成标准包（`lib/cmake/ncnn/ncnnConfig.cmake` + `ncnn.cmake` 导出）。
2. oracle 生成的工程 `find_package(ncnn)` + `target_link_libraries(runner ncnn)`，传递依赖全自动带入。

> 注意：**不能直接用 build 树的 `ncnnConfig.cmake`**——它 `include` 的 `ncnn.cmake` 导出文件只在
> `install` 时生成。故必须先 install 到前缀。oracle 的 `ncnn_dir` 指向 `build_lib_vk/install/lib/cmake/ncnn`。

---

## 5. 实现中踩平的 3 个真实坑

### 坑 1：链接缺导出文件
build 树 `ncnnConfig.cmake` 引用了 install 期才生成的 `ncnn.cmake`，`find_package` 报错。
→ 改为先 `cmake --install` 到前缀，oracle 指向 install 包。

### 坑 2：workgroup 维度不匹配（"处理一半"现象）
首次 GPU 跑 `max_diff=4.65`，且只有**每 16 元素的前 8 个**被处理。根因：候选用了默认
`set_optimal_local_size_xyz()` = 3D `(4,4,4)`，与 1D dispatch（`dispatcher.w=n, h=1, c=1`）错位。
→ 照搬内建 `sigmoid_vulkan`，改 1D 工作组 `set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1)`。

### 坑 3：record_upload 自动 pack 成 elempack=4
调试打印发现上传后 `w=16 h=2 c=1 elempack=4 cstep=32`——`record_upload` 在 pack 维度能被 4 整除时
**强制 pack 成 elempack=4**（`command.cpp`：`dst_elempack = elemcount%4==0 ? 4 : 1`），且不受
`use_packing_layout` 控制。标量 shader 只摸到了 1/4 数据。
→ v1 在 runner 里 `vkdev->convert_packing(in_gpu, tmp, 1, cmd, opt)` 强制回 elempack=1，保持候选 shader 简单（标量）。

附带：shader 声明了 `layout(constant_id=0) const int n`，故 `Pipeline::create` 的 specialization
向量长度必须 = 1（设为 0，让 `psc(n)` 回落到 push-constant 的动态值），否则
"specialization count mismatch"。

---

## 6. 环境配置（Apple M5）

- **编译** vulkan libncnn + runner：本机即可（`NCNN_SIMPLEVK=ON` 自带 vulkan 头 + 运行时 dlopen，
  glslang bundled），**不需要 Vulkan SDK**。
- **运行/验证**：需 Vulkan ICD。已 `brew install molten-vk`（`/opt/homebrew/lib/libMoltenVK.dylib`）。
  ncnn simplevk 读 `NCNN_VULKAN_DRIVER` 直接 dlopen ICD（MoltenVK 导出 `vk_icdGetInstanceProcAddr`），
  **无需完整 Vulkan loader**。`VulkanLayerOracle` 在 macOS 自动探测 MoltenVK 设此变量，开箱即用。
- 一次性前置：`git -C ncnn submodule update --init glslang`（glslang 是 submodule，非源码自带）。
- 项目侧建了 `.venv`（cmake + numpy）。

### 一次性构建命令
```bash
git -C ncnn submodule update --init glslang
cmake -S ncnn -B ncnn/build_lib_vk -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ncnn/build_lib_vk -j8
cmake --install ncnn/build_lib_vk --prefix ncnn/build_lib_vk/install
```

---

## 7. 已验证结果

```bash
python opgen/layer_oracle/run_vulkan_oracle.py
```

| 场景 | 结果 |
|---|---|
| 有 MoltenVK | `COMPILE OK` + `PASS  max_diff=0.000000`（`Cand_AbsVal_vulkan` vs `np.abs`，Apple M5 GPU） |
| 无 GPU 驱动 | `COMPILE OK` + `SKIPPED: no vulkan device`（优雅跳过，非 fail） |
| 不设环境变量 | 自动探测 MoltenVK → `PASS`（开箱即用） |

COMPILE gate 本身即证明"vulkan 编译/链接通路打通"；PASS 进一步证明 GPU 数值正确。

---

## 8. v1 边界

- 仅 elempack=1（runner 强制 unpack）；NC4HW4 的 GPU 端 packed shader 路径留待后续。
- 样例为 elementwise（AbsVal）；带权重 / 多输入路径已写但未跨样例验证。
- 自检脚本用 numpy 参考；走 KernelAgent 时用 PyTorch oracle（与 base/arm 一致）。
- 仍缺：bridge 永久安装 `.comp`（仅最终 register/整网集成需要；隔离验证用运行时编 shader，不需要）。

---

## 9. 生成端：vulkan 算子的 prompt 与 agent 通路（已接）

KernelAgent 现在能端到端**生成 + 验证** vulkan 算子，与 base/arm 同构。

### 9.1 prompt（核心）

`kernel_prompts.py: VULKAN_LAYER_BACKGROUND` —— 教 LLM 写**三件套**并避开所有已知坑：
- **三文件**：`cand_<op>_vulkan.{h,cpp}` + 独立 `cand_<op>.comp`（算法在 shader 里）。
- **结构**：子类 `: public Cand_<Op>`；构造里 **`support_vulkan = true`（强制，否则 oracle 拒绝）**；`create_pipeline` 用 `compile_candidate_shader(opt, spirv)`（helper）运行时编 shader → `pipeline->create(spv, ...)`；`forward_inplace(VkMat&, VkCompute&, opt)` 用 `record_pipeline` dispatch。
- **明确禁令**：不引用 `LayerShaderType`（构建期枚举不可用）；不写 `DEFINE_LAYER_CREATOR`；v1 用标量 `sfp`/`buffer_ld1`（非 `sfpvec4`/pack4）。
- **防坑硬约束**：`set_optimal_local_size_xyz(subgroup_size, 1, 1)`（1D，否则只处理部分数据）；specialization 向量长度 = shader `constant_id` 数；shader 用 ncnn 方言（`psc(n)` push-constant）。
- `_files_instruction` 让 coder/debugger 都明确"输出三文件"；`debugger_prompt` 加了 vulkan 专属修复提示（漏 `support_vulkan`、3D workgroup、spec 数不符、引用 LayerShaderType 等）。

### 9.2 通路接线

- `kernel_schemas.py: KernelProfile.shader` 字段；`as_backend("vulkan")` 派生 `Cand_<Op>_vulkan` + `cand_<op>.comp`（复用 base 的 params/weights/flags，无需二次 analyzer）。
- `kernel_agent.py`：`backend=="vulkan"` → 用 `VulkanLayerOracle`、从 base profile 派生（同 arm 走 `_subclasses_base`）、把已验证 base 文件作 `base_files` 编入。
- `kernel_pipeline.py: verify_kernel`：按 backend 组 `backend_kwargs`——vulkan 传 `shader=`（base/arm 传 `packing=`）；无 GPU 时 `verdict.skipped` → 记为 `numeric_skipped`（编译已验，不算失败）。
- `cli/run_kernel_agent.py`：`--backend vulkan`（像 arm 一样先加载 base kernel）。

用法：`python opgen/cli/run_kernel_agent.py --task Exp --backend vulkan`（需先有 base run；运行需 Vulkan 设备，无则 SKIP）。

---

## 附：关键代码位置

| 主题 | 位置 |
|---|---|
| 运行时编译 shader | `ncnn/src/gpu.cpp: compile_spirv_module`（`gpu.h:556`） |
| 裸 SPIR-V 建管线 | `ncnn/src/pipeline.h:30 Pipeline::create(spv_data,...)` |
| 隔离 vulkan 跑法蓝图 | `ncnn/tests/testutil.cpp:895-1175` |
| 自动 pack elempack 逻辑 | `ncnn/src/command.cpp: VkCompute::record_upload`（`dst_elempack = elemcount%4==0?4:1`） |
| psc 宏定义 | `ncnn/src/gpu.cpp:5476` `psc(x)=(x==0?p.x:x)` |
| 多后端选择 + 静默回退 | `ncnn/src/layer.cpp: create_layer / Layer_final` |
| simplevk ICD 加载（NCNN_VULKAN_DRIVER） | `ncnn/src/simplevk.cpp:110, 278-320` |
| 本项目 vulkan oracle | `opgen/layer_oracle/vulkan_oracle.py` / `vulkan_oracle_runner.cpp` / `cand_vulkan_shader.h` |
| 样例算子 | `opgen/layer_oracle/samples/cand_absval_vulkan.{h,cpp,comp}` |
| 自检 | `opgen/layer_oracle/run_vulkan_oracle.py` |
| vulkan 生成 prompt | `opgen/kernel/kernel_prompts.py: VULKAN_LAYER_BACKGROUND / _background / _files_instruction` |
| vulkan agent 通路 | `kernel_agent.py`（oracle 分派、`as_backend("vulkan")`）/ `kernel_pipeline.py: verify_kernel`（shader 路由、skip）/ `kernel_schemas.py: KernelProfile.shader` / `cli/run_kernel_agent.py: --backend vulkan` |
