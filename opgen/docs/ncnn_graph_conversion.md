# 为 ncnn 框架新增算子的「计算图转换」流程

> 背景：之前的 MoKA Agent 只能为 ncnn **已有**的算子写/改 kernel 实现（`ncnn/src/layer/*.cpp/h`），但无法引入框架里**不存在**的全新算子。原因是——一个移动端算子要真正可用，除了 kernel 实现，还必须有一条**从 PyTorch 模型 → ncnn 计算图**的转换链路，把 PyTorch 的 op 映射成 ncnn 的 layer + param + weight。本文梳理这条链路：要实现哪些文件、跑哪些测试、如何验证转换正确。

本文基于 `ncnn/tools/` 实测内容整理，转换工具的现代主路径是 **PNNX**（`tools/pnnx`），`tools/onnx/onnx2ncnn` 已标记为 legacy。

---

## 0. 一个算子「完整可用」需要两层东西

| 层 | 产物 | 谁负责 | 在哪 |
|---|---|---|---|
| **运行时 kernel 层** | `Layer` 子类：`load_param` / `load_model` / `forward` | MoKA 已能做 | `ncnn/src/layer/<op>.cpp/.h` + `src/layer_registry.h` |
| **计算图转换层** | PyTorch op → ncnn layer 的图改写 pass + param/weight 映射 | **本文要补的能力** | `ncnn/tools/pnnx/src/...` |

只有 kernel、没有转换层 → PyTorch 模型里的这个 op 无法被翻译进 `.ncnn.param`，benchmark/correctness 拿不到模型；
只有转换层、没有 kernel → ncnn 加载时找不到 layer 实现，运行报错。
**新增一个全新算子必须两层都补齐。** 本文聚焦转换层（PNNX）。

---

## 1. PNNX 转换的整体链路

```
model.pt (TorchScript)
   │  pnnx model.pt inputshape=[...]
   ▼
[load_torchscript]  → 原始 TorchScript 图（aten::* / prim::*）
   ▼
pass_level0   图预处理（inline、常量折叠、shape 推断…）
   ▼
pass_level1   把 nn.Module 捕获成高层算子          ← FuseModulePass（如 nn.Hardsigmoid）
   ▼
pass_level2   把 aten::*/prim:: 子图匹配成 PNNX 算子  ← GraphRewriterPass（如 F.hardsigmoid）
   ▼
pass_level3/4/5  融合、化简、形状传播、优化
   ▼
   ├── save 出 PNNX 自有格式：model.pnnx.param / .bin / _pnnx.py / .pnnx.onnx
   │
   ▼
pass_ncnn     把 PNNX 算子改写成 ncnn layer + 填 param/weight  ← GraphRewriterPass（ncnn 命名空间）
   ▼
[save_ncnn]   → model.ncnn.param / model.ncnn.bin / model_ncnn.py
```

一次 `pnnx model.pt inputshape=[1,3,224,224]` 通常产出 7 个文件：
```
*.pnnx.param  *.pnnx.bin  *_pnnx.py  *.pnnx.onnx        ← PNNX 中间表示
*.ncnn.param  *.ncnn.bin  *_ncnn.py                     ← ncnn 最终产物（MoKA benchmark 用的就是这两个）
```

> 关键认知：PyTorch op 要落到 ncnn，必须在 **pass_level1/2**（torch → pnnx）和 **pass_ncnn**（pnnx → ncnn）这两段都有对应 pass。新增算子的工作量主要在这两处。

---

## 2. 目录与文件职责（tools/pnnx/src）

```
src/
 ├── main.cpp                入口，串起所有 pass
 ├── load_torchscript.cpp    .pt → 内部图
 ├── load_onnx.cpp           .onnx → 内部图（pnnx 也能吃 onnx）
 ├── ir.h / ir.cpp           IR 定义：Graph / Operator / Operand / Parameter / Attribute
 ├── pass_level0/            图预处理 pass
 ├── pass_level1/            ★ nn.Module 捕获（FuseModulePass）  —— 每个 nn.XXX 一个 .cpp
 ├── pass_level2/            ★ aten/prim 子图 → PNNX 算子（GraphRewriterPass）—— 215 个文件
 ├── pass_level3/4/5/        融合化简优化
 ├── pass_ncnn/              ★★ PNNX 算子 → ncnn layer（GraphRewriterPass, ncnn 命名空间）
 │     ├── F_*.cpp           functional 类算子（F.hardsigmoid, F.conv2d…）
 │     ├── nn_*.cpp          带权重的 module 算子（nn.Conv2d, nn.Linear… 73 个）
 │     ├── Tensor_*.cpp      张量操作（reshape/permute/expand…）
 │     ├── convert_*.cpp     特殊转换（slice/cat/chunk/custom_op/module_op…）
 │     └── ...
 ├── save_ncnn.cpp           输出 .ncnn.param / .ncnn.bin
 └── save_pnnx.cpp           输出 pnnx 格式
```

三类 pass 的注册宏：
- `REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CLASS)`             —— pass_level1
- `REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(CLASS, PRIO)`    —— pass_level2 / pass_level5（torch 域）
- `REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(CLASS, PRIO)` —— pass_ncnn

---

## 3. PNNX IR 文本格式（看懂 match_pattern）

pass 用一段 PNNX-IR 文本描述要匹配/替换的子图。以 hardsigmoid 为例：
```
7767517            ← magic
3 2                ← 算子数=3, 操作数(operand)数=2
pnnx.Input    input  0 1 input          类型  名字  入边数 出边数  操作数名...
F.hardsigmoid op_0   1 1 input out      [可带 key=value 参数 / @weight @bias 权重占位]
pnnx.Output   output 1 0 out
```
- 每行：`类型  名字  输入数  输出数  输入操作数...  输出操作数...  [参数...]`
- `%name`：捕获该参数到 `captured_params`（如 `out_channels=%out_channels`）。
- `@weight @bias`：捕获权重到 `captured_attrs`。
- `%*=%*`：通配所有参数（常用于 onnx 来源）。

---

## 4. 新增一个算子要实现哪些文件

按算子的来源形态，分别需要补不同的 pass。下面给出"全新算子" `MyOp` 的完整清单。

### 4.1 torch → pnnx（pass_level1 / pass_level2）

**情况 A：算子来自一个 `nn.Module`（如自定义 `nn.MyOp`）** → 在 `pass_level1/` 加一个 `FuseModulePass`：
```cpp
// pass_level1/nn_MyOp.cpp
class MyOp : public FuseModulePass {
public:
    const char* match_type_str() const { return "__torch__.torch.nn.modules.xxx.MyOp"; }
    const char* type_str() const { return "nn.MyOp"; }
};
REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MyOp)
```

**情况 B：算子来自 functional / aten 调用（如 `F.my_op` 或 `torch.my_op`）** → 在 `pass_level2/` 加一个 `GraphRewriterPass`，把 `aten::my_op` 子图收成一个 PNNX 算子 `F.my_op`：
```cpp
// pass_level2/F_my_op.cpp
class F_my_op : public GraphRewriterPass {
public:
    const char* match_pattern_graph() const {
        return R"PNNXIR(7767517
3 2
pnnx.Input   input  0 1 input
aten::my_op  op_0   1 1 input out
pnnx.Output  output 1 0 out
)PNNXIR";
    }
    const char* type_str() const { return "F.my_op"; }
};
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_my_op, 100)
```

> **重点：一个 PyTorch op 在不同导出路径下会产生不同子图**，所以经常要写多个匹配模式。看 `pass_level2/F_hardsigmoid.cpp` 就有 ~10 个变体：
> - `F_hardsigmoid`：直接 `aten::hardsigmoid`
> - `F_hardsigmoid_2/_2_1/_2_2`：被展开成 `add + clamp + div/mul` 的组合
> - `F_hardsigmoid_3/_4/_5`：`hardtanh` / `relu6` / `nn.ReLU6` 变体
> - `F_hardsigmoid_onnx/_onnx_1/_2/_3`：onnx 导出路径的 `HardSigmoid` 节点，且用 `match()` 校验 alpha/beta，用 `replace_pattern_graph()` 在参数非标准时插入 `mul/add` 还原语义
>
> 这正是"新增算子难"的核心：要覆盖该 op 的各种导出展开形态，否则换个写法就匹配不到。

### 4.2 pnnx → ncnn（pass_ncnn）★ 最关键

在 `pass_ncnn/` 加一个 `GraphRewriterPass`（ncnn 命名空间），把 PNNX 算子翻译成 ncnn layer 并填好 param/weight：
```cpp
// pass_ncnn/F_my_op.cpp
namespace pnnx { namespace ncnn {
class F_my_op : public GraphRewriterPass {
public:
    const char* match_pattern_graph() const {
        return R"PNNXIR(7767517
3 2
pnnx.Input  input 0 1 input
F.my_op     op_0  1 1 input out
pnnx.Output output 1 0 out
)PNNXIR";
    }
    const char* type_str() const { return "MyOp"; }   // ← ncnn 的 layer 类型名（须与 src/layer 注册一致）
    const char* name_str() const { return "myop"; }   // ← 实例名前缀
    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const {
        op->params["0"] = ...;   // ← 把参数按 ncnn 该 layer 的 param id 填进去
    }
};
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_my_op, 20)
}} // namespace
```

**带权重的算子**（如 conv/linear/bn）用 `write(op, captured_params, captured_attrs)`，参考 `pass_ncnn/nn_Conv2d.cpp`：
- `op->params["0"] = out_channels; op->params["1"] = kernel_w; ...` 按 ncnn Convolution 的 param 编号逐个填；
- `op->params["6"] = captured_attrs.at("op_0.weight").elemcount();` 权重元素数；
- `op->attrs["0"]` 写量化标志位 `{0,0,0,0}`，`op->attrs["1"] = weight`，`op->attrs["2"] = bias`；
- 复杂情形用 `replace_pattern_graph()` 把一个 PNNX 算子展开成**多个** ncnn layer（如 `padding_mode!=zeros` 时 Conv2d → `Padding + Convolution` 两层），并在 `match()` 里做条件判定（同一 op 注册多个优先级不同的变体 `nn_Conv2d / _1 / _2 / _3`）。

> ncnn 各 layer 的 param 编号、weight 布局规范见：
> `ncnn/docs/developer-guide/operation-param-weight-table.md` 与 `param-and-model-file-structure.md`。
> 这是 `write()` 里 `params["N"]` 数字含义的**权威对照表**，新增算子必须照它填。

### 4.3 运行时 kernel（与 MoKA 衔接）

转换层把 op 写成 ncnn layer 类型 `MyOp` 后，ncnn 加载时需要有对应的 `Layer` 实现。这一步就是 MoKA 原本擅长的：在 `ncnn/src/layer/` 加 `myop.cpp/.h` 并在构建系统注册（`add_layer`），实现 `forward`。详见 `docs/developer-guide/how-to-implement-custom-layer-step-by-step.md`。

### 4.4 新增算子改动文件清单（速查）

| 文件 | 必需性 | 作用 |
|---|---|---|
| `pass_level1/nn_MyOp.cpp` | 算子是 nn.Module 时 | 捕获 module → `nn.MyOp` |
| `pass_level2/F_my_op.cpp`（可能多变体） | functional/aten 时 | aten 子图 → `F.my_op` |
| `pass_ncnn/F_my_op.cpp` 或 `nn_MyOp.cpp` | **必需** | PNNX 算子 → ncnn layer + param/weight |
| `src/layer/myop.cpp/.h` + 构建注册 | **必需** | ncnn 运行时 kernel（MoKA 负责） |
| `tools/pnnx/tests/ncnn/test_F_my_op.py` | 强烈建议 | 端到端正确性测试 |
| `CMakeLists.txt`（pnnx src + tests/ncnn） | 必需 | 把新 .cpp 编进 pnnx、把测试登记进 ctest |

---

## 5. 怎么测试 / 验证转换正确

### 5.1 测试结构

```
tools/pnnx/tests/         torch → pnnx 转换测试（验证 pnnx 自有格式）
tools/pnnx/tests/ncnn/    ★ torch → ncnn 端到端测试（验证 pass_ncnn + ncnn 推理）
tools/pnnx/tests/onnx/    onnx 来源转换测试
tools/pnnx/tests/run_test.cmake   统一 runner（执行 py 脚本，返回码非 0 即失败）
```

### 5.2 端到端测试脚本套路（以 `tests/ncnn/test_F_hardsigmoid.py` 为例）

```python
class Model(nn.Module):
    def forward(self, x, y, z, w):
        # 故意覆盖多种 rank（1D/2D/3D/4D）和多种等价写法
        x = F.hardsigmoid(x)
        z = F.relu6(z + 3.) / 6.      # 等价展开形态，测匹配覆盖
        ...

def test():
    net = Model().eval()
    x = torch.rand(16); y = torch.rand(2,16); z = torch.rand(3,12,16); w = torch.rand(5,7,9,11)
    a = net(x, y, z, w)                                   # ① PyTorch 参考输出

    mod = torch.jit.trace(net, (x,y,z,w)); mod.save("test_F_hardsigmoid.pt")  # ② 导出 TorchScript
    os.system("../../src/pnnx test_F_hardsigmoid.pt inputshape=[16],[2,16],[3,12,16],[5,7,9,11]")  # ③ 转换

    import test_F_hardsigmoid_ncnn                        # ④ pnnx 生成的 ncnn python 推理脚本
    b = test_F_hardsigmoid_ncnn.test_inference()          #    跑 ncnn 推理

    for a0, b0 in zip(a, b):                              # ⑤ 逐输出比对
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True
```

验证逻辑 = **PyTorch 原始输出  vs  转换成 ncnn 后再推理的输出**，用 `torch.allclose(rtol=1e-4, atol=1e-4)` 判等。
这同时验证了三件事：① pass_level1/2 把 torch op 收成了 PNNX 算子；② pass_ncnn 正确翻译成 ncnn layer + 参数；③ ncnn kernel 实现数值正确。

> 注意：测试脚本会刻意写多种 rank 和多种等价表达（`F.hardsigmoid` / `relu6(x+3)/6` / 自定义 forward），用来确保 pass_level2 的各匹配变体都被命中——新增算子写测试时也应如此覆盖。

### 5.3 注册与运行

测试在 `tests/ncnn/CMakeLists.txt` 用 `pnnx_ncnn_add_test(F_my_op)` 登记，落到 ctest：
```cmake
macro(pnnx_ncnn_add_test name)
    add_test(NAME test_ncnn_${name}
        COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=... -DPYTHON_SCRIPT=.../test_${name}.py
                -P .../run_test.cmake)
endmacro()
pnnx_ncnn_add_test(F_my_op)
```
运行：
```bash
# 构建 pnnx（需先装好 libtorch / torchvision c++）
cd ncnn/tools/pnnx && mkdir build && cd build && cmake .. && make -j8
# 跑全部 ncnn 转换测试
ctest -R test_ncnn_                # 或单测：ctest -R test_ncnn_F_my_op -V
```
`run_test.cmake` 执行 py 脚本，返回码非 0 → `FATAL_ERROR`，即测试失败。

### 5.4 手工 / 调试验证

- **看转换结果是否人类可读正确**：转换后直接看 `model.ncnn.param`（文本格式，每行一个 layer），确认新算子被翻成了期望的 ncnn layer 类型与参数；也可把 `model.pnnx.param` 拖进 https://netron.app 可视化。
- **定位匹配失败**：若某 op 没被转换，多半是 pass_level2 没覆盖该导出形态——对照 `*.pnnx.param` 里残留的 `aten::*/prim::*` 节点补匹配模式。
- **参数填错**：对照 `operation-param-weight-table.md` 核对 `params["N"]` 编号与 weight 布局。

---

## 6. 与 MoKA Agent 的衔接（为何要把这条链路纳入）

MoKA 原 NCNN pipeline 的 register/compile/verify/benchmark 都建立在"该算子已能转出 `.ncnn.param`"之上：
- `correctness` 用 `ncnn/build/tests_naive/<test>` 这种已存在的测试 exe；
- `benchmark` 用 `dataset/Mobilekernelbench_pt_ncnn_success/.../<task>.ncnn.param`（**这正是 pnnx 转换的产物**）。

→ 对一个框架里**全新**的算子，上述前提不成立：没有转换 pass 就生不成 `.ncnn.param`，整条 pipeline 第一步就断。
因此端到端 Agent 必须新增一个**「计算图转换」能力 / 阶段**，让 Agent 能：
1. 生成/修改 `pass_level1/2` 与 `pass_ncnn` 的转换 pass（C++）；
2. 重新编译 pnnx；
3. 用 `tests/ncnn/test_*.py` 的套路（PyTorch 参考 vs ncnn 推理 + allclose）**自验证转换正确**；
4. 转换成功产出 `.ncnn.param/.bin` 后，再进入原有的 kernel 实现 / 编译 / 正确性 / 性能流程。

这条转换链路（pass_level1/2 + pass_ncnn + tests/ncnn 自验证）就是端到端 Agent 相对旧 MoKA 需要补齐的核心新增功能。

---

## 7. 一句话总结

> 为 ncnn 新增一个全新算子 = **两层 + 两段转换 + 一套端到端自验证**：
> 运行时补 `src/layer/<op>` kernel；转换层在 `pass_level1/2`（torch→pnnx，常需多变体覆盖导出形态）和 `pass_ncnn`（pnnx→ncnn，按 param/weight 表填参数、必要时一拆多）各写 GraphRewriterPass；
> 用 `tests/ncnn/test_*.py` 以「PyTorch 输出 vs 转 ncnn 后推理输出 `allclose`」端到端验证；
> 通过后 `pnnx model.pt inputshape=...` 即可稳定产出 `.ncnn.param/.bin`，下游 MoKA 的编译/正确性/性能流程才得以接续。
